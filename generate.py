import argparse
import os
import random
import time 
from datetime import datetime
import json
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

def get_lora_display_name(lora_path):
    if not lora_path:
        return "base"
    
    parts = [p for p in lora_path.rstrip(os.sep).split(os.sep) if p]
    if not parts:
        return "lora"

    # Handle 'version_X' folders by climbing up one level
    if parts[-1].startswith("version_") and len(parts) > 1:
        return parts[-2]

    # If the last part is a generic name, climb up the directory tree

    if (parts[-1] in ["lora", "best_lora", "checkpoints"] or parts[-1].startswith("checkpoint-")) and len(parts) > 1:
        if parts[-2].startswith("version_") and len(parts) > 2:
            return parts[-3]
        else:
            return parts[-2]
    
    return parts[-1]


def resolve_lora_path(args):
    if not args.lora:
        return None

    selected_lora = args.lora


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
            return None
        
        selected_lora = random.choice(all_loras)
    

    # This now runs for both "random" and specific inputs like "super-mario-rpg"
    selected_name = os.path.basename(selected_lora.rstrip(os.sep))
    if selected_name in ["lora", "best_lora", "checkpoints"] or selected_name.startswith("checkpoint-"):
        parts = selected_lora.rstrip(os.sep).split(os.sep)
        if len(parts) > 1:
            selected_name = parts[-2]

    version = args.version
    

    if version and version.lower() == "random":
        log_base = os.path.join("lightning_logs", selected_name)
        if os.path.exists(log_base):
            versions = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d)) and d.startswith("version_")]
            if versions:
                version = random.choice(versions)
    
    potential_paths = []
    

    search_roots = []
    if version and version != "random":
        search_roots.append(os.path.join("lightning_logs", selected_name, version))

        if version.isdigit():
             search_roots.append(os.path.join("lightning_logs", selected_name, f"version_{version}"))
    
    search_roots.extend([
        os.path.join("models", "loras", selected_name),
        os.path.join("models", selected_name),
        os.path.join("lightning_logs", selected_name),
    ])


    for root in search_roots:

        if args.step:
             potential_paths.append(os.path.join(root, f"checkpoint-{args.step}"))

        # Fallback/Default paths
        potential_paths.append(os.path.join(root, "lora"))
        potential_paths.append(root)
    
    # Also check the direct path provided in args.lora (in case it was a direct path to a checkpoint)
    if args.step:
         potential_paths.append(os.path.join(selected_lora, f"checkpoint-{args.step}"))

    potential_paths.append(selected_lora)
    
    for p in potential_paths:
        if os.path.exists(p) and (os.path.isdir(p) or p.endswith(".safetensors")):
            return p
            
    # If nothing resolved, return the original input so error handling can catch it later
    return selected_lora

def main():
    parser = argparse.ArgumentParser(description="Unified MIDI Generator")

    # Model Args
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--config", type=str, default="auto", help="Model config")
    parser.add_argument("--lora", type=str, default=None, help="LoRA path or 'random'")

    # parser.add_argument("--best_lora", action="store_true", help="Use best_lora if available") 

    parser.add_argument("--step", type=str, default=None, help="Specific LoRA step (e.g., 50, 100)")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="Strength of LoRA")
    parser.add_argument("--version", type=str, default=None, help="Version or 'random'")
    parser.add_argument("--loop", type=float, nargs='?', const=0, default=None, help="Keep base model in memory and loop. Optionally provide sleep seconds.")

    # Input Args
    parser.add_argument("--input", type=str, default=None)
    # ‼️ Added input_bars argument to specify how many bars of the input to keep
    parser.add_argument("--input_bars", type=float, default=None, help="Amount of bars to take from the beginning of the input")
    # ‼️ Added input_start_bar argument to specify where to start taking bars from
    parser.add_argument("--input_start_bar", type=float, default=0.0, help="Start point in bars for the input slice")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--instruments", type=str, nargs="+")
    parser.add_argument("--bpm", type=str, default="0")
    parser.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures)
    parser.add_argument("--time_sig", type=str, default="auto")

    # Generation Constraints
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num_events", type=int, default=512)
    parser.add_argument("--num_bars", type=float, default=None)
    parser.add_argument("--merge_output", action="store_true")

    # Sampling
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=20)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.config == "auto":
        config_path = os.path.join(os.path.dirname(args.model), "config.json")
        config = MIDIModelConfig.from_json_file(config_path) if os.path.exists(config_path) else MIDIModelConfig.from_name("tv2o-medium")
    else:
        config = MIDIModelConfig.from_json_file(args.config) if os.path.exists(args.config) else MIDIModelConfig.from_name(args.config)

    print(f"Loading Base Model Weights: {args.model}")
    if args.model.endswith(".safetensors"):
        base_state_dict = safe_load_file(args.model)
    else:
        ckpt = torch.load(args.model, map_location="cpu")
        base_state_dict = ckpt.get("state_dict", ckpt)

    tokenizer = config.tokenizer


    model = MIDIModel(config=config)

    while True:

        # This effectively "resets" any previous LoRA applications
        model.load_state_dict(base_state_dict, strict=False)


        current_lora_path = resolve_lora_path(args)
        
        if args.lora and args.lora.lower() == "random":
             print(f"🎲 Selected Random LoRA: {current_lora_path}")
        elif current_lora_path and current_lora_path != args.lora:
             print(f"✅ Resolved LoRA: {current_lora_path}")

        if current_lora_path:
            print(f"Merging LoRA: {current_lora_path}")



            meta_path = None
            if os.path.isdir(current_lora_path):
                # Check current dir (old style)
                if os.path.exists(os.path.join(current_lora_path, "metadata.json")):
                    meta_path = os.path.join(current_lora_path, "metadata.json")
                # Check parent dir (new style, where current_lora_path might be 'checkpoint-50')
                elif os.path.exists(os.path.join(os.path.dirname(current_lora_path), "best_loss_info.json")):
                    meta_path = os.path.join(os.path.dirname(current_lora_path), "best_loss_info.json")
            
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                        print(f"   ℹ️ Best Loss for this run was at Epoch {meta.get('epoch', '?')}, Step {meta.get('step', '?')} (Loss: {meta.get('value', '?')})")
                except Exception as e:
                    pass

            try:
                model = model.load_merge_lora(current_lora_path, lora_scale=args.lora_strength)
            except Exception as e:
                print(f"⚠️ Failed to merge LoRA: {e}")

        model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

        mid_np = None
        disable_patch_change = False
        disable_channels = []
        full_mid_tokens = []

        if args.input:
            with open(args.input, 'rb') as f:
                mid_tokens = tokenizer.tokenize(MIDI.midi2score(f.read()), cc_eps=4, tempo_eps=4, remap_track_channel=True, add_default_instr=True)

            if mid_tokens and mid_tokens[-1][0] == tokenizer.eos_id: mid_tokens = mid_tokens[:-1]
            
            # ‼️ Logic to slice input (start bar and duration)
            if args.input_bars is not None or args.input_start_bar > 0:
                # Assuming 4/4 time signature (4 beats per bar)
                start_beats = args.input_start_bar * 4
                
                # If input_bars is provided, calculate end beats. Otherwise infinite.
                duration_beats = (args.input_bars * 4) if args.input_bars is not None else float('inf')
                end_beats = start_beats + duration_beats
                
                accumulated_beats = 0
                
                start_idx = None
                end_idx = len(mid_tokens)
                
                # Get ID for time1=0
                time1_base = tokenizer.parameter_ids["time1"][0]

                for i, event_tokens in enumerate(mid_tokens):
                    # Always skip BOS for timing calc
                    if event_tokens[0] == tokenizer.bos_id:
                        continue
                    
                    if len(event_tokens) < 2:
                        continue

                    t1_id = event_tokens[1]
                    # Check if it's a valid time1 parameter ID
                    if t1_id < time1_base:
                        continue
                        
                    t1_val = t1_id - time1_base
                    accumulated_beats += t1_val

                    # Check if we passed the start threshold
                    if start_idx is None and accumulated_beats >= start_beats:
                        start_idx = i
                    
                    # Check if we passed the end threshold
                    if accumulated_beats >= end_beats:
                        end_idx = i
                        print(f"‼️ Truncating input: Bars {args.input_start_bar} to {args.input_start_bar + (args.input_bars if args.input_bars else '?')}")
                        break
                
                # Handle cases where start wasn't found (e.g. start at 0)
                if start_idx is None:
                    if args.input_start_bar == 0:
                        start_idx = 1 # Start after BOS
                    else:
                         print("⚠️ Warning: Start bar exceeds file length.")
                         start_idx = len(mid_tokens)

                # Construct new token list: BOS + Slice
                bos_token = mid_tokens[0]
                sliced_content = mid_tokens[start_idx:end_idx]
                mid_tokens = [bos_token] + sliced_content

            full_mid_tokens = list(mid_tokens)
            mid_np = np.asarray([mid_tokens] * args.batch_size, dtype=np.int64)
        else:
            mid_prompt = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
            

            current_bpm = args.bpm
            if current_bpm == "random":
                current_bpm = random.choice([70, 80, 90, 100, 110, 120, 130, 140])
            
            try:
                bpm_val = int(current_bpm)
                if bpm_val > 0:
                    # ["set_tempo", time1, time2, track, bpm]
                    mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm_val]))
                    print(f"🎵 BPM set to: {bpm_val}")
            except ValueError:
                pass


            current_ts = args.time_sig
            if current_ts == "random":
                current_ts = random.choice(["4/4", "3/4", "6/8"])
            
            if current_ts != "auto" and current_ts is not None:
                try:
                    nn, dd = map(int, current_ts.split('/'))
                    # ["time_signature", time1, time2, track, nn, dd]
                    mid_prompt.append(tokenizer.event2tokens(["time_signature", 0, 0, 0, nn-1, int(np.log2(dd))]))
                    print(f"🎼 Time Signature set to: {current_ts}")
                except Exception as e:
                    print(f"⚠️ Failed to set time sig: {e}")

            if args.instruments:
                patches = {i: patch2number[instr] for i, instr in enumerate(args.instruments) if instr in patch2number}
                for i, (c, p) in enumerate(patches.items()):
                    mid_prompt.append(tokenizer.event2tokens(["patch_change", 0, 0, i + 1, c, p]))
                disable_patch_change = True
                disable_channels = [i for i in range(16) if i not in patches]
            mid_np = np.asarray([mid_prompt] * args.batch_size, dtype=np.int64)

        seed = args.seed if args.seed is not None else random.randint(0, MAX_SEED)
        print(f"🌱 Seed: {seed}")
        generator = torch.Generator(device).manual_seed(seed)
        
        output_tokens = model.generate(
            prompt=mid_np, batch_size=args.batch_size, max_len=mid_np.shape[1] + args.num_events,
            temp=args.temp, top_p=args.top_p, top_k=args.top_k, generator=generator,
            disable_patch_change=disable_patch_change, disable_channels=disable_channels
        )

        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        lora_display_name = get_lora_display_name(current_lora_path)

        if args.step:
            lora_display_name += f"_step{args.step}"
        
        for i in range(args.batch_size):
            tokens = output_tokens[i]
            mid_score = tokenizer.detokenize(tokens.tolist())
            
            if args.output:
                base_output = args.output
            else:
                base_output = f"{lora_display_name}_{timestamp}.mid"

            if args.batch_size > 1:
                root, ext = os.path.splitext(base_output)
                fname = os.path.join("output", f"{root}_{i}{ext}")
            else:
                fname = os.path.join("output", base_output)
                
            with open(fname, 'wb') as f:
                f.write(MIDI.score2midi(mid_score))
            print(f"✅ Saved: {fname}")

        if args.loop is None:
            break
        
        if args.loop > 0:
            print(f"--- ‼️ Generation finished. Sleeping for {args.loop}s... ---")
            time.sleep(args.loop)
        else:
            print("--- ‼️ Iteration finished. Looping... ---")

        # No need to del model, we reuse it.
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
