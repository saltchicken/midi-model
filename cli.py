import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from safetensors.torch import load_file as safe_load_file


import MIDI
from midi_model import MIDIModel, MIDIModelConfig
from midi_tokenizer import MIDITokenizer


number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = ['C♭', 'A♭m', 'G♭', 'E♭m', 'D♭', 'B♭m', 'A♭', 'Fm', 'E♭', 'Cm', 'B♭', 'Gm', 'F', 'Dm',
                  'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F♯m', 'E', 'C♯m', 'B', 'G♯m', 'F♯', 'D♯m', 'C♯', 'A♯m']

MAX_SEED = np.iinfo(np.int32).max


@torch.inference_mode()
def generate(model, tokenizer, prompt=None, batch_size=1, max_len=512, temp=1.0, top_p=0.98, top_k=20,
             disable_patch_change=False, disable_control_change=False, disable_channels=None, generator=None):
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    
    max_token_seq = tokenizer.max_token_seq
    
    # Initialize input tensor (Bos + Pad or Prompt)
    if prompt is None:
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
        input_tensor[0, 0] = tokenizer.bos_id
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
    else:
        if len(prompt.shape) == 2:
            prompt = prompt[None, :]
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif prompt.shape[0] == 1:
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
            raise ValueError(f"invalid shape for prompt, {prompt.shape}")
        
        prompt = prompt[..., :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(prompt, ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                            mode="constant", constant_values=tokenizer.pad_id)
        input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
    
    # Keep only last 4096 events context
    input_tensor = input_tensor[:, -4096:]
    cur_len = input_tensor.shape[1]
    

    bar = tqdm.tqdm(desc="Generating", total=max_len - cur_len)
    
    from transformers import DynamicCache
    cache1 = DynamicCache()
    past_len = 0
    
    with bar:
        while cur_len < max_len:
            end = [False] * batch_size
            hidden = model.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
            next_token_seq = None
            event_names = [""] * batch_size
            cache2 = DynamicCache()
            
            for i in range(max_token_seq):
                mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.int64, device=model.device)
                for b in range(batch_size):
                    if end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    if i == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                        if disable_patch_change:
                            mask_ids.remove(tokenizer.event_ids["patch_change"])
                        if disable_control_change:
                            mask_ids.remove(tokenizer.event_ids["control_change"])
                        mask[b, mask_ids] = 1
                    else:
                        param_names = tokenizer.events[event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        param_name = param_names[i - 1]
                        mask_ids = tokenizer.parameter_ids[param_name]
                        if param_name == "channel":
                            mask_ids = [i for i in mask_ids if i not in disable_channels]
                        mask[b, mask_ids] = 1
                
                mask = mask.unsqueeze(1)
                x = next_token_seq
                if i != 0:
                    hidden = None
                    x = x[:, -1:]
                
                logits = model.forward_token(hidden, x, cache=cache2)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                samples = model.sample_top_p_k(scores, top_p, top_k, generator=generator)
                
                if i == 0:
                    next_token_seq = samples
                    for b in range(batch_size):
                        if end[b]:
                            continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id:
                            end[b] = True
                        else:
                            event_names[b] = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                    if all([len(tokenizer.events[event_names[b]]) == i for b in range(batch_size) if not end[b]]):
                        break
            
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                       "constant", value=tokenizer.pad_id)
            
            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            past_len = cur_len
            cur_len += 1
            bar.update(1)
            
            if all(end):
                break
                
    return input_tensor.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="MIDI Model CLI Generator")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model file (.ckpt or .safetensors)")
    parser.add_argument("--config", type=str, default="auto", help="Model config name (e.g. tv2o-medium) or path to config.json")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter folder or huggingface id")
    parser.add_argument("--output", type=str, default="output.mid", help="Output MIDI filename")
    parser.add_argument("--num_events", type=int, default=512, help="Max MIDI events to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of files to generate")
    parser.add_argument("--verbose", action="store_true", help="Print generated events to console")
    
    # Prompt Options
    parser.add_argument("--input_midi", type=str, help="Input MIDI file to continue from")
    parser.add_argument("--instruments", type=str, nargs="+", help="List of instruments (e.g. 'Acoustic Grand', 'Violin')")
    parser.add_argument("--bpm", type=int, default=0, help="BPM (0 for auto)")
    parser.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures, help="Key signature")
    parser.add_argument("--time_sig", type=str, default="auto", help="Time signature (e.g. 4/4)")
    
    # Gen Options
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.98, help="Top P")
    parser.add_argument("--top_k", type=int, default=20, help="Top K")
    parser.add_argument("--lora-layer", type=int, default=None, help="LoRA Layer") # ‼️
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA Alpha") # ‼️
    
    args = parser.parse_args()

    # 1. Setup Device & Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config loading logic from app.py
    if args.config == "auto":
        config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        if os.path.exists(config_path):
            config = MIDIModelConfig.from_json_file(config_path)
        else:
            print("❌ Config not found automatically. Use --config to specify.")
            return
    else:
        if os.path.exists(args.config):
            config = MIDIModelConfig.from_json_file(args.config)
        else:
            config = MIDIModelConfig.from_name(args.config)

    print(f"Loading model from {args.model_path}...")
    model = MIDIModel(config=config)
    tokenizer = model.tokenizer

    # Load weights
    if args.model_path.endswith(".safetensors"):
        state_dict = safe_load_file(args.model_path)
    else:
        ckpt = torch.load(args.model_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    
    model.load_state_dict(state_dict, strict=False)


    if args.lora_path:
        print(f"Loading and merging LoRA from {args.lora_path}...")
        model = model.load_merge_lora(args.lora_path, layer=args.lora_layer, alpha=args.lora_alpha) # ‼️

    model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

    # 2. Prepare Prompt
    seed = args.seed if args.seed is not None else np.random.randint(0, MAX_SEED)
    print(f"Seed: {seed}")
    generator = torch.Generator(device).manual_seed(seed)
    
    mid_np = None
    disable_channels = []
    disable_patch_change = False

    if args.input_midi:
        print(f"Loading prompt from {args.input_midi}...")
        with open(args.input_midi, 'rb') as f:
            midi_data = f.read()
        mid_score = MIDI.midi2score(midi_data)
        # Tokenize existing MIDI
        mid_tokens = tokenizer.tokenize(mid_score, cc_eps=4, tempo_eps=4, 
                                        remap_track_channel=True, add_default_instr=True, remove_empty_channels=False)
        mid_tokens = mid_tokens[:4096] # Truncate if too long
        mid_np = np.asarray([mid_tokens] * args.batch_size, dtype=np.int64)
    else:
        # Build custom prompt from args
        print("Constructing custom prompt...")
        mid_list = [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)
        mid_prompt = [mid_list]


        forced_time_sig = False
        forced_bpm = False

        if tokenizer.version == "v2":
            if args.time_sig != "auto":
                nn, dd = args.time_sig.split('/')
                dd_map = {2: 1, 4: 2, 8: 3}
                mid_prompt.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, int(nn)-1, dd_map.get(int(dd), 2)]))

            elif args.instruments:
                print("ℹ️  Auto-selecting 4/4 Time Signature")
                mid_prompt.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, 3, 2])) # 4/4
            
            if args.key_sig != "auto":
                idx = key_signatures.index(args.key_sig)
                key_sig_sf = idx // 2 - 7
                key_sig_mi = idx % 2
                mid_prompt.append(tokenizer.event2tokens(["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]))

        if args.bpm != 0:
            mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, args.bpm]))

        elif args.instruments:
            print("ℹ️  Auto-selecting 120 BPM")
            mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, 120]))

        patches = {}
        idx = 0
        if args.instruments:
            for instr in args.instruments:
                if instr in patch2number:
                    patches[idx] = patch2number[instr]
                    idx = (idx + 1) if idx != 8 else 10
            
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]

        for i, (c, p) in enumerate(patches.items()):
            mid_prompt.append(tokenizer.event2tokens(["patch_change", 0, 0, i + 1, c, p]))

        mid_np = np.asarray([mid_prompt] * args.batch_size, dtype=np.int64)

    # 3. Run Generation
    current_prompt_len = mid_np.shape[1]
    total_len = current_prompt_len + args.num_events
    
    output_tokens = generate(
        model, tokenizer, prompt=mid_np, batch_size=args.batch_size, max_len=total_len,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k,
        disable_patch_change=disable_patch_change, disable_channels=disable_channels, generator=generator
    )

    # 4. Save Output
    for i in range(args.batch_size):
        tokens = output_tokens[i]
        seq = tokens.tolist()
        mid_score = tokenizer.detokenize(seq)
        

        if args.verbose:
            print(f"\n--- Events for Batch {i} ---")
            for t_seq in seq:
                event = tokenizer.tokens2event(t_seq)
                if event:
                    print(event)

        fname = args.output
        if args.batch_size > 1:
            base, ext = os.path.splitext(fname)
            fname = f"{base}_{i}{ext}"
            
        with open(fname, 'wb') as f:
            f.write(MIDI.score2midi(mid_score))
        print(f"Saved: {fname}")

if __name__ == "__main__":
    main()
