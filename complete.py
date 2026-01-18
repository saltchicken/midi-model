import argparse
import os
import random
import numpy as np
import torch
import MIDI
from midi_model import MIDIModel, MIDIModelConfig
from safetensors.torch import load_file as safe_load_file
from cli import generate  # Re-use the generation logic from your existing CLI

def main():
    parser = argparse.ArgumentParser(description="Feed in a MIDI file and have it completed by the model.")
    
    # Required Arguments
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input MIDI file to continue.")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to the model checkpoint (.ckpt or .safetensors).")
    
    # Optional Arguments
    parser.add_argument("--output", "-o", type=str, default="completed.mid", help="Path to save the completed MIDI.")
    parser.add_argument("--config", "-c", type=str, default="auto", help="Model config name (e.g. tv2o-medium) or path to config.json.")
    parser.add_argument("--num_events", "-n", type=int, default=512, help="Number of MIDI events to generate.")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter if using one.")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="Strength of LoRA.")
    
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
        print(f"MERGING LORA: {args.lora} (Strength: {args.lora_strength})")
        model = model.load_merge_lora(args.lora, lora_scale=args.lora_strength)

    model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()

    # 5. Process Input MIDI
    print(f"READING PROMPT: {args.input}")
    with open(args.input, 'rb') as f:
        midi_data = f.read()
    
    mid_score = MIDI.midi2score(midi_data)
    mid_tokens = tokenizer.tokenize(mid_score, cc_eps=4, tempo_eps=4, 
                                    remap_track_channel=True, add_default_instr=True, remove_empty_channels=False)

    # ‼️ CRITICAL: Smart Truncation
    # We want to keep the END of the file so the model continues it.
    # We also need to leave room for the new events (args.num_events).
    max_context = 4096 # Standard model limit
    safe_prompt_len = max_context - args.num_events - 16 # Buffer

    if len(mid_tokens) > safe_prompt_len:
        print(f"⚠️ Input too long ({len(mid_tokens)} tokens). Keeping last {safe_prompt_len} tokens.")
        mid_tokens = mid_tokens[-safe_prompt_len:]
    
    mid_np = np.asarray([mid_tokens], dtype=np.int64)

    # 6. Generate
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"SEED: {seed}")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device).manual_seed(seed)

    total_len = mid_np.shape[1] + args.num_events
    print(f"GENERATING: {args.num_events} events (Total context: {total_len})...")

    output_tokens = generate(
        model, tokenizer, prompt=mid_np, batch_size=1, max_len=total_len,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k, 
        generator=generator
    )

    # 7. Save Output
    print(f"SAVING: {args.output}")
    mid_score_out = tokenizer.detokenize(output_tokens[0].tolist())
    with open(args.output, 'wb') as f:
        f.write(MIDI.score2midi(mid_score_out))
    print("Done!")
