import argparse
import os
import random
from datetime import datetime
import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file

import MIDI
from midi_model import MIDIModel, MIDIModelConfig

# Constants from cli.py
number2drum_kits = {-1: "None", 0: "Standard", 8: "Room", 16: "Power", 24: "Electric", 25: "TR-808", 32: "Jazz",
                    40: "Blush", 48: "Orchestra"}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
key_signatures = ['C♭', 'A♭m', 'G♭', 'E♭m', 'D♭', 'B♭m', 'A♭', 'Fm', 'E♭', 'Cm', 'B♭', 'Gm', 'F', 'Dm',
                  'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F♯m', 'E', 'C♯m', 'B', 'G♯m', 'F♯', 'D♯m', 'C♯', 'A♯m']
MAX_SEED = np.iinfo(np.int32).max

def main():
    parser = argparse.ArgumentParser(description="Unified MIDI Generator (Generation & Completion)")

    # Model Args
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.ckpt or .safetensors)")
    parser.add_argument("--config", type=str, default="auto", help="Model config name or path")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter or 'random'")
    parser.add_argument("--best_lora", action="store_true", help="Use best_lora directory if available")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="Strength of LoRA")
    parser.add_argument("--version", type=str, default=None, help="Lightning logs version (e.g. version_0) or 'random'")
    
    # Input Args (Dual Mode)
    parser.add_argument("--input", type=str, default=None, help="Input MIDI file. If provided, acts as completion mode.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of files to generate")
    
    # Prompt Construction Args (Used if --input is NOT provided)
    parser.add_argument("--instruments", type=str, nargs="+", help="List of instruments (e.g. 'Acoustic Grand')")
    parser.add_argument("--bpm", type=str, default="0", help="BPM (0 for auto, or 'random')")
    parser.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures, help="Key signature")
    parser.add_argument("--time_sig", type=str, default="auto", help="Time signature (e.g. 4/4 or 'random')")

    # Segmentation Args (Used if --input IS provided)
    parser.add_argument("--segment_mode", choices=["start", "end"], default="end", help="Context from start or end of input")
    parser.add_argument("--segment_limit", type=int, default=None, help="Number of events for context")
    parser.add_argument("--segment_bars", type=float, default=None, help="Number of bars for context")
    
    # Generation Constraints
    parser.add_argument("--output", type=str, default=None, help="Output filename/path (If None, uses [lora]_[timestamp].mid)")
    parser.add_argument("--num_events", type=int, default=512, help="Max events to generate")
    parser.add_argument("--num_bars", type=float, default=None, help="Max bars to generate (overrides num_events)")
    parser.add_argument("--merge_output", action="store_true", help="Append generation to original input file")

    # Sampling
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.config == "auto":
        config_path = os.path.join(os.path.dirname(args.model), "config.json")
        if os.path.exists(config_path):
            config = MIDIModelConfig.from_json_file(config_path)
        else:
            config = MIDIModelConfig.from_name("tv2o-medium")
            print(f"⚠️ Config defaulting to tv2o-medium")
    else:
        if os.path.exists(args.config):
            config = MIDIModelConfig.from_json_file(args.config)
        else:
            config = MIDIModelConfig.from_name(args.config)

    print(f"Loading model: {args.model}")
    model = MIDIModel(config=config)
    tokenizer = model.tokenizer

    if args.model.endswith(".safetensors"):
        state_dict = safe_load_file(args.model)
    else:
        ckpt = torch.load(args.model, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    
    model.load_state_dict(state_dict, strict=False)

    if args.lora:
        if args.lora.lower() == "random":
            all_loras = []
            scan_roots = ["models/loras", "models", "lightning_logs"]
            
            for root_dir in scan_roots:
                if not os.path.exists(root_dir):
                    continue
                for root, dirs, files in os.walk(root_dir):
                    if "adapter_config.json" in files:
                        all_loras.append(root)
            
            if not all_loras:
                print("❌ No LoRAs found to pick from randomly.")
                return
            
            args.lora = random.choice(all_loras)
            print(f"🎲 Randomly selected LoRA Path: {args.lora}")


        if args.version and args.version.lower() == "random":
            log_base = os.path.join("lightning_logs", os.path.basename(args.lora.rstrip(os.sep)))
            if os.path.exists(log_base):
                versions = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d)) and d.startswith("version_")]
                if versions:
                    args.version = random.choice(versions)
                    print(f"🎲 Randomly selected Version: {args.version}")
                else:
                    args.version = None
                    print("⚠️ No version folders found, using base LoRA path.")
            else:
                args.version = None
                print(f"⚠️ Log base {log_base} not found, using base LoRA path.")

        potential_paths = []
        
        # 1. Try deriving best_lora from input path if flag is set
        if args.best_lora:
            if args.lora.rstrip(os.sep).endswith("lora"):
                    parent = os.path.dirname(args.lora.rstrip(os.sep))
                    potential_paths.append(os.path.join(parent, "best_lora"))
            potential_paths.append(os.path.join(args.lora, "best_lora"))

        # 2. Add standard search locations
        search_roots = []
        if args.version:
                search_roots.append(os.path.join("lightning_logs", args.lora, args.version))
        
        search_roots.extend([
            os.path.join("models", "loras", args.lora),
            os.path.join("models", args.lora),
            os.path.join("lightning_logs", args.lora),
        ])

        for root in search_roots:
            if args.best_lora:
                potential_paths.append(os.path.join(root, "best_lora"))
            potential_paths.append(os.path.join(root, "lora"))
            potential_paths.append(root)
        
        # 3. Handle the direct argument (fallback or primary)
        if not args.best_lora:
             potential_paths.insert(0, args.lora)
        else:
             potential_paths.append(args.lora)
        
        lora_path = args.lora 
        for p in potential_paths:
            if os.path.exists(p) and (os.path.isdir(p) or p.endswith(".safetensors")):
                print(f"‼️ Found LoRA at {p}")
                lora_path = p
                break
        
        print(f"Loading LoRA: {lora_path}")
        model = model.load_merge_lora(lora_path, lora_scale=args.lora_strength)

    model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

    # 2. Prepare Prompt (Dual Strategy)
    disable_patch_change = False
    disable_channels = []
    full_mid_tokens = [] # For merging later

    # Strategy A: Input File (Completion Mode)
    if args.input:
        print(f"Reading input: {args.input}")
        with open(args.input, 'rb') as f:
            midi_data = f.read()
        mid_score = MIDI.midi2score(midi_data)
        mid_tokens = tokenizer.tokenize(mid_score, cc_eps=4, tempo_eps=4, 
                                        remap_track_channel=True, add_default_instr=True, remove_empty_channels=False)
        
        if mid_tokens and mid_tokens[-1][0] == tokenizer.eos_id:
            mid_tokens = mid_tokens[:-1]
        
        full_mid_tokens = list(mid_tokens)


        # Calculate Segment Limits
        max_context = 4096 
        safe_len = max_context - 16 
        
        # Filter by BARS
        if args.segment_bars is not None:
            target_beats = args.segment_bars * 4 
            print(f"Filtering input by {args.segment_bars} bars")

            def get_event_beats(tokens):
                 if len(tokens) > 1 and tokens[0] != tokenizer.bos_id:
                     val = tokens[1] - tokenizer.parameter_ids["time1"][0]
                     return max(0, val)
                 return 0

            if args.segment_mode == "start":
                accumulated_beats = 0
                split_index = 0
                for i, t in enumerate(mid_tokens):
                    accumulated_beats += get_event_beats(t)
                    split_index = i + 1
                    if accumulated_beats >= target_beats:
                        break
                mid_tokens = mid_tokens[:split_index]
            else: # end
                total_beats = sum(get_event_beats(t) for t in mid_tokens)
                start_beat_threshold = max(0, total_beats - target_beats)
                current_beat = 0
                split_index = 0
                for i, t in enumerate(mid_tokens):
                    current_beat += get_event_beats(t)
                    if current_beat >= start_beat_threshold:
                        split_index = i
                        break
                
                mid_tokens = mid_tokens[split_index:]
                # Ensure BOS if needed
                if mid_tokens and mid_tokens[0][0] != tokenizer.bos_id and full_mid_tokens[0][0] == tokenizer.bos_id:
                     mid_tokens = [full_mid_tokens[0]] + mid_tokens

        # Filter by EVENTS
        else:
            limit = safe_len
            if args.segment_limit and args.segment_limit > 0:
                limit = min(args.segment_limit, safe_len)
            
            if len(mid_tokens) > limit:
                if args.segment_mode == "start":
                    mid_tokens = mid_tokens[:limit]
                else:
                    # Take end, preserve BOS
                    if mid_tokens and mid_tokens[0][0] == tokenizer.bos_id:
                        mid_tokens = [mid_tokens[0]] + mid_tokens[-(limit-1):]
                    else:
                        mid_tokens = [mid_tokens[0]] + mid_tokens[-limit:]

        mid_np = np.asarray([mid_tokens] * args.batch_size, dtype=np.int64)

    # Strategy B: Construct from Args (Scratch Mode)
    else:
        print("Constructing prompt from arguments...")
        mid_list = [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)
        mid_prompt = [mid_list]

        if tokenizer.version == "v2":

            if args.time_sig.lower() == "random":
                args.time_sig = random.choice(["4/4", "3/4", "2/4", "6/8", "5/4"])
                print(f"🎲 Randomly selected Time Signature: {args.time_sig}")

            if args.time_sig != "auto":
                nn, dd = args.time_sig.split('/')
                dd_map = {2: 1, 4: 2, 8: 3}
                mid_prompt.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, int(nn)-1, dd_map.get(int(dd), 2)]))
            elif args.instruments:
                mid_prompt.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, 3, 2])) # 4/4
            
            # Key Sig
            if args.key_sig != "auto":
                idx = key_signatures.index(args.key_sig)
                key_sig_sf = idx // 2 - 7
                key_sig_mi = idx % 2
                mid_prompt.append(tokenizer.event2tokens(["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]))


        if str(args.bpm).lower() == "random":
            bpm_val = random.randint(60, 200)
            print(f"🎲 Randomly selected BPM: {bpm_val}")
        else:
            bpm_val = int(args.bpm)

        if bpm_val != 0:
            mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm_val]))
        elif args.instruments:
            mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, 120]))

        # Instruments
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


    # 3. Generate
    seed = args.seed if args.seed is not None else random.randint(0, MAX_SEED)
    print(f"Seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    generator = torch.Generator(device).manual_seed(seed)

    # Determine constraints
    stop_events = args.num_events
    if args.num_bars is not None:
        stop_events = 2048 # Safety buffer
    
    current_prompt_len = mid_np.shape[1]
    total_len = current_prompt_len + stop_events

    print(f"Generating...")
    output_tokens = model.generate(
        prompt=mid_np, 
        batch_size=args.batch_size, 
        max_len=total_len,
        temp=args.temp, 
        top_p=args.top_p, 
        top_k=args.top_k,
        disable_patch_change=disable_patch_change,
        disable_channels=disable_channels,
        stop_bars=args.num_bars,
        stop_events=stop_events if args.num_bars is None else None,
        generator=generator
    )

    # 4. Save
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle output filename ‼️
    if args.output:
        base_output = args.output
    else:
        # Determine lora name from path if available ‼️
        if args.lora:
            # Split the path and filter out generic names like 'lora' or 'best_lora' ‼️
            parts = [p for p in args.lora.rstrip(os.sep).split(os.sep) if p]
            if parts:
                # If the last part is a generic name, climb up the directory tree ‼️
                if parts[-1] in ["lora", "best_lora", "checkpoints"] and len(parts) > 1:
                    # If the parent is a 'version_X', try to go one level higher ‼️
                    if parts[-2].startswith("version_") and len(parts) > 2:
                        lora_name = parts[-3]
                    else:
                        lora_name = parts[-2]
                else:
                    lora_name = parts[-1]
            else:
                lora_name = "lora"
        else:
            lora_name = "base"
            
        base_output = f"{lora_name}_{timestamp}.mid"

    if os.path.dirname(base_output):
        os.makedirs(os.path.dirname(base_output), exist_ok=True)
    else:
        base_output = os.path.join("output", base_output)

    for i in range(args.batch_size):
        tokens = output_tokens[i]
        
        # Merge if requested (only works if input was provided)
        if args.merge_output and args.input:
            prompt_len = mid_np.shape[1]
            generated_events = tokens[prompt_len:]
            final_seq = full_mid_tokens + generated_events.tolist()
            mid_score = tokenizer.detokenize(final_seq)
        else:
            mid_score = tokenizer.detokenize(tokens.tolist())
            
        # Unique filename per batch ‼️
        if args.batch_size > 1:
            root, ext = os.path.splitext(base_output)
            fname = f"{root}_{i}{ext}"
        else:
            fname = base_output
            
        with open(fname, 'wb') as f:
            f.write(MIDI.score2midi(mid_score))
        print(f"Saved: {fname}")

if __name__ == "__main__":
    main()
