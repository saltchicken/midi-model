import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import DynamicCache
import MIDI
from midi_model import MIDIModel, MIDIModelConfig
from safetensors.torch import load_file as safe_load_file


# from cli import generate 


@torch.inference_mode()
def generate_custom(model, tokenizer, prompt=None, batch_size=1, max_len=512, 
                    stop_bars=None, stop_events=None, 
                    temp=1.0, top_p=0.98, top_k=20, generator=None):
    
    max_token_seq = tokenizer.max_token_seq
    device = model.device

    # Initialize input tensor
    if prompt is None:
        input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=device)
        input_tensor[0, 0] = tokenizer.bos_id
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
    else:
        # Handle numpy to tensor conversion
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt).to(dtype=torch.long, device=device)
            
        if len(prompt.shape) == 2:
            prompt = prompt[None, :]
            prompt = prompt.repeat(batch_size, 1, 1)
        elif prompt.shape[0] == 1:
            prompt = prompt.repeat(batch_size, 1, 1)
            
        prompt = prompt[..., :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = F.pad(prompt, (0, max_token_seq - prompt.shape[-1]), "constant", tokenizer.pad_id)
        input_tensor = prompt

    # Keep context window
    input_tensor = input_tensor[:, -4096:]
    cur_len = input_tensor.shape[1]

    # Tracking metrics
    accumulated_beats = 0.0
    generated_events_count = 0
    target_beats = stop_bars * 4 if stop_bars else None # Assuming 4/4 time

    bar = tqdm.tqdm(desc="Generating", total=stop_events if stop_events else int(target_beats) if target_beats else (max_len - cur_len))
    
    cache1 = DynamicCache()
    past_len = 0
    
    with bar:
        while cur_len < max_len:
            end = [False] * batch_size
            
            # Forward pass for next event prediction
            hidden = model.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
            next_token_seq = None
            event_names = [""] * batch_size
            cache2 = DynamicCache()
            
            # Generate tokens for the single event
            for i in range(max_token_seq):
                mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.int64, device=device)
                for b in range(batch_size):
                    if end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    if i == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                        mask[b, mask_ids] = 1
                    else:
                        param_names = tokenizer.events[event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        param_name = param_names[i - 1]
                        mask_ids = tokenizer.parameter_ids[param_name]
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
                        if end[b]: continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id:
                            end[b] = True
                        else:
                            event_names[b] = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                    if all([len(tokenizer.events[event_names[b]]) == i for b in range(batch_size) if not end[b]]):
                        break
            
            # Pad if necessary
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]), "constant", value=tokenizer.pad_id)
            
            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            
            # Update state
            past_len = cur_len
            cur_len += 1
            generated_events_count += 1
            

            if not end[0]:
                # Extract time1 token (index 1) from the generated event (batch index 0)
                # Structure: [EventID, Time1, Time2, ...]
                t1_id = next_token_seq[0, 0, 1].item()
                # Offset check to be safe
                if t1_id >= tokenizer.parameter_ids["time1"][0]:
                    t1_val = t1_id - tokenizer.parameter_ids["time1"][0]
                    accumulated_beats += t1_val

            # Update Progress Bar
            if target_beats:
                 bar.n = min(int(accumulated_beats), int(target_beats))
                 bar.refresh()
            else:
                 bar.update(1)

            # Check Stops
            if all(end):
                break
            
            if target_beats and accumulated_beats >= target_beats:
                break
            
            if stop_events and generated_events_count >= stop_events:
                break
                
    return input_tensor.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Feed in a MIDI file and have it completed by the model.")
    
    # Required Arguments
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input MIDI file to continue.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model checkpoint (.ckpt or .safetensors).")
    
    # Optional Arguments
    parser.add_argument("--output", "-o", type=str, default="completed.mid", help="Path to save the completed MIDI.")
    parser.add_argument("--config", "-c", type=str, default="auto", help="Model config name (e.g. tv2o-medium) or path to config.json.")
    parser.add_argument("--num_events", "-n", type=int, default=512, help="Max number of MIDI events to generate (safety limit if num_bars is set).")

    parser.add_argument("--num_bars", type=float, default=None, help="Number of bars to generate (overrides num_events).")
    
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter if using one.")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="Strength of LoRA.")
    
    # Segment Arguments
    parser.add_argument("--segment_mode", choices=["start", "end"], default="end", help="Whether to take context from the start or end of the input file.")
    parser.add_argument("--segment_limit", type=int, default=None, help="Number of events to use for context (default: max available).")
    parser.add_argument("--segment_bars", type=float, default=None, help="Number of bars to use for context (approx 4 beats/bar). Overrides segment_limit.")
    parser.add_argument("--merge_output", action="store_true", help="If set, appends the generated segments to the FULL original input file.")

    # Generation Parameters
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature (creativity).")
    parser.add_argument("--top_p", type=float, default=0.98, help="Top P sampling.")
    parser.add_argument("--top_k", type=int, default=20, help="Top K sampling.")

    args = parser.parse_args()

    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {device}")

    # 2. Load Config
    if args.config == "auto":
        config_path = os.path.join(os.path.dirname(args.model), "config.json")
        if os.path.exists(config_path):
            config = MIDIModelConfig.from_json_file(config_path)
        else:
            print("⚠️ Config not found automatically. Defaulting to 'tv2o-medium'. Use --config to specify.")
            config = MIDIModelConfig.from_name("tv2o-medium")
    else:
        if os.path.exists(args.config):
            config = MIDIModelConfig.from_json_file(args.config)
        else:
            config = MIDIModelConfig.from_name(args.config)

    # 3. Load Model
    print(f"LOADING: {args.model}")
    model = MIDIModel(config=config)
    tokenizer = model.tokenizer

    if args.model.endswith(".safetensors"):
        state_dict = safe_load_file(args.model)
    else:
        ckpt = torch.load(args.model, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    
    model.load_state_dict(state_dict, strict=False)

    # 4. Load LoRA (Optional)
    if args.lora:
        lora_path = args.lora
        if not os.path.exists(args.lora):
            potential_path = os.path.join("models", "loras", args.lora)
            if os.path.exists(potential_path):
                print(f"Found LoRA at {potential_path}")
                lora_path = potential_path
        
        print(f"MERGING LORA: {lora_path} (Strength: {args.lora_strength})")
        model = model.load_merge_lora(lora_path, lora_scale=args.lora_strength)

    model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

    # 5. Process Input MIDI
    print(f"READING PROMPT: {args.input}")
    with open(args.input, 'rb') as f:
        midi_data = f.read()
    
    mid_score = MIDI.midi2score(midi_data)
    mid_tokens = tokenizer.tokenize(mid_score, cc_eps=4, tempo_eps=4, 
                                    remap_track_channel=True, add_default_instr=True, remove_empty_channels=False)

    if mid_tokens and mid_tokens[-1][0] == tokenizer.eos_id:
        mid_tokens = mid_tokens[:-1]

    # Save FULL original tokens 
    full_mid_tokens = list(mid_tokens) 

    # Calculate Segment Limits
    max_context = 4096 
    safe_len = max_context - 16 # Just buffer, num_events is dynamic now
    
    # Check if we are filtering by BARS or EVENTS
    if args.segment_bars is not None:
        target_beats = args.segment_bars * 4 
        print(f"‼️ Filtering input by {args.segment_bars} bars (~{target_beats} beats)")

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
            
            if mid_tokens and mid_tokens[0][0] == tokenizer.bos_id:
                bos_event = mid_tokens[0]
                content_tokens = mid_tokens[split_index:] 
                if split_index == 0: pass 
                elif content_tokens and content_tokens[0][0] == tokenizer.bos_id:
                    mid_tokens = content_tokens
                else:
                    mid_tokens = [bos_event] + content_tokens
            else:
                 mid_tokens = mid_tokens[split_index:]

    else:
        # Filter by Event Count
        limit = safe_len # Default to max safe length if not specified
        if args.segment_limit and args.segment_limit > 0:
            limit = min(args.segment_limit, safe_len)
        
        if len(mid_tokens) > limit:
            if args.segment_mode == "start":
                print(f"‼️ Segment Mode: START | Taking first {limit} events.")
                mid_tokens = mid_tokens[:limit]
            else:
                print(f"‼️ Segment Mode: END | Taking last {limit} events.")
                if mid_tokens and mid_tokens[0][0] == tokenizer.bos_id:
                    bos_event = mid_tokens[0]
                    content_tokens = mid_tokens[1:]
                    content_tokens = content_tokens[-(limit - 1):]
                    mid_tokens = [bos_event] + content_tokens
                else:
                    mid_tokens = mid_tokens[-limit:]
    
    mid_np = np.asarray([mid_tokens], dtype=np.int64)

    # 6. Generate
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"SEED: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device).manual_seed(seed)


    stop_bars = args.num_bars
    stop_events = args.num_events if stop_bars is None else 2048 # High limit if bars are set
    
    total_len = mid_np.shape[1] + stop_events
    
    print(f"GENERATING: {'Target ' + str(stop_bars) + ' bars' if stop_bars else str(stop_events) + ' events'}...")


    output_tokens = generate_custom(
        model, tokenizer, prompt=mid_np, batch_size=1, max_len=total_len,
        stop_bars=stop_bars, stop_events=stop_events,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k, 
        generator=generator
    )

    # 7. Save Output
    # Ensure output is put in the output folder if no directory specified ‼️
    if not os.path.dirname(args.output): # ‼️
        os.makedirs("output", exist_ok=True) # ‼️
        output_path = os.path.join("output", args.output) # ‼️
    else:
        output_path = args.output
    
    output_path = os.path.abspath(output_path)
    print(f"SAVING: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.merge_output:
        print("‼️ Merging generated events with the original full input file...")
        prompt_len = mid_np.shape[1]
        generated_events = output_tokens[0][prompt_len:]
        generated_events_list = generated_events.tolist()
        final_token_sequence = full_mid_tokens + generated_events_list
        mid_score_out = tokenizer.detokenize(final_token_sequence)
    else:
        mid_score_out = tokenizer.detokenize(output_tokens[0].tolist())

    with open(output_path, 'wb') as f:
        f.write(MIDI.score2midi(mid_score_out))
    print(f"Done! File saved successfully at: {output_path}")

if __name__ == "__main__":
    main()
