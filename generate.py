import argparse
import os
import random
import sys
import queue
import threading
import subprocess
import time
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

# ‼️ Helper to calculate duration for seamless looping ‼️
def calculate_chunk_duration(score, current_tempo):
    ticks_per_beat = score[0]
    max_tick = 0
    for track in score[1:]:
        for event in track:
            # event structure: [name, start_tick, ...]
            # note structure: [name, start_tick, duration, ...]
            start = event[1]
            duration = 0
            if event[0] == 'note':
                duration = event[2]
            
            end_tick = start + duration
            if end_tick > max_tick:
                max_tick = end_tick
    
    # Duration = (ticks / ticks_per_beat) * (microseconds_per_beat / 1,000,000)
    # We use current_tempo as an approximation for the chunk
    seconds = (max_tick / ticks_per_beat) * (current_tempo / 1000000.0)
    return seconds

# ‼️ Helper to trim the end of the generation to avoid cadence/resolution artifacts ‼️
def trim_tokens(tokenizer, tokens, n_events_to_trim=8):
    # ‼️ FIX: Simply slice. Python handles slicing larger than list length gracefully (returns empty).
    # Previous logic returned the FULL list if it was short, which is bad (plays the ending).
    if n_events_to_trim <= 0: return tokens
    return tokens[:-n_events_to_trim]

def main():
    parser = argparse.ArgumentParser(description="Unified MIDI Generator (Generation & Completion)")

    # Model Args
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.ckpt or .safetensors)")
    parser.add_argument("--config", type=str, default="auto", help="Model config name or path")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--lora_strength", type=float, default=1.0, help="Strength of LoRA")
    parser.add_argument("--version", type=str, default=None, help="Lightning logs version (e.g. version_0)")
    
    # Input Args (Dual Mode)
    parser.add_argument("--input", type=str, default=None, help="Input MIDI file. If provided, acts as completion mode.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of files to generate")
    
    # Prompt Construction Args (Used if --input is NOT provided)
    parser.add_argument("--instruments", type=str, nargs="+", help="List of instruments (e.g. 'Acoustic Grand')")
    parser.add_argument("--bpm", type=int, default=0, help="BPM (0 for auto)")
    parser.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures, help="Key signature")
    parser.add_argument("--time_sig", type=str, default="auto", help="Time signature (e.g. 4/4)")

    # Segmentation Args (Used if --input IS provided)
    parser.add_argument("--segment_mode", choices=["start", "end"], default="end", help="Context from start or end of input")
    parser.add_argument("--segment_limit", type=int, default=None, help="Number of events for context")
    parser.add_argument("--segment_bars", type=float, default=None, help="Number of bars for context")
    
    # Generation Constraints
    parser.add_argument("--output", type=str, default="output.mid", help="Output filename/path")
    parser.add_argument("--num_events", type=int, default=512, help="Max events to generate")
    parser.add_argument("--num_bars", type=float, default=None, help="Max bars to generate (overrides num_events)")
    parser.add_argument("--merge_output", action="store_true", help="Append generation to original input file")

    # Sampling
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")


    parser.add_argument("--infinite", action="store_true", help="Generate and play music indefinitely")
    parser.add_argument("--port", type=str, default=None, help="MIDI output port for infinite mode (e.g. '128:0'). Run 'aplaymidi -l' to find.")

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

        lora_path = args.lora
        if not os.path.exists(lora_path):
            potential_paths = []
            

            if args.version:
                 potential_paths.append(os.path.join("lightning_logs", args.lora, args.version, "lora"))

            potential_paths.extend([
                os.path.join("models", "loras", args.lora),
                os.path.join("models", args.lora),
                os.path.join("lightning_logs", args.lora),
            ])
            
            for p in potential_paths:
                if os.path.exists(p):
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
                        mid_tokens = mid_tokens[-limit:]

        mid_np = np.asarray([mid_tokens] * args.batch_size, dtype=np.int64)

    # Strategy B: Construct from Args (Scratch Mode - from cli.py)
    else:
        print("Constructing prompt from arguments...")
        mid_list = [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)
        mid_prompt = [mid_list]

        if tokenizer.version == "v2":
            # Time Sig
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

        # BPM
        if args.bpm != 0:
            mid_prompt.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, args.bpm]))
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


    if args.infinite:
        if args.output != "output.mid":
             print("⚠️ Output file ignored in infinite mode.")
        
        # Check for aplaymidi
        try:
            subprocess.run(["aplaymidi", "-V"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("❌ 'aplaymidi' not found. Please install alsa-utils (e.g., 'sudo pacman -S alsa-utils').")
            return

        # Port Selection logic
        if not args.port:
            print("🔍 Scanning MIDI ports...")
            result = subprocess.run(["aplaymidi", "-l"], capture_output=True, text=True)
            print(result.stdout)
            print("❌ Please specify a port using --port (e.g., --port 128:0)")
            print("   (Tip: You need a MIDI synth running, like 'fluidsynth' or 'timidity -iA')")
            return

        print(f"🎵 Starting Infinite Generation on port {args.port}...")
        print("   (Press Ctrl+C to stop)")
        
        # Queue for playback
        # ‼️ Increased buffer size to 16 (was 8) to buffer ahead and avoid pauses
        playback_queue = queue.Queue(maxsize=16)
        
        # ‼️ Shared state for graceful shutdown ‼️
        stop_event = threading.Event()
        process_lock = threading.Lock()
        
        # ‼️ Track active processes to clean up zombies
        active_processes = []
        
        # ‼️ Track the player thread so we can start it lazily
        player_thread = None

        # Worker thread to play MIDI while model generates next chunk
        def player_worker():
            # ‼️ Track underrun state to avoid spamming ‼️
            underrun_flag = False

            while not stop_event.is_set():
                # ‼️ Clean up finished processes to avoid zombies
                for p in active_processes[:]:
                     if p.poll() is not None:
                          active_processes.remove(p)

                try:
                    # ‼️ Use timeout so we can check stop_event frequently ‼️
                    # ‼️ Queue now returns (bytes, duration)
                    item = playback_queue.get(timeout=0.1)
                    midi_data, duration = item
                    
                    # ‼️ If we recovered from an underrun, notify user ‼️
                    if underrun_flag:
                          print(f"\n▶️ Buffer recovered. Resuming playback...    ")
                          underrun_flag = False

                except queue.Empty:
                    # ‼️ If queue is empty, generation is lagging behind playback ‼️
                    if not underrun_flag and not stop_event.is_set():
                        print(f"\n⚠️ Buffer empty! Silence is due to generation lag...", end="\r")
                        sys.stdout.flush()
                        underrun_flag = True
                    continue
                
                if midi_data is None: break
                
                try:
                    # Play (Non-Blocking)
                    # ‼️ Use Popen and write to stdin, but DO NOT wait for it immediately
                    with process_lock:
                        if stop_event.is_set(): break
                        proc = subprocess.Popen(
                            ["aplaymidi", "-p", args.port, "-"], 
                            stdin=subprocess.PIPE
                        )
                        active_processes.append(proc)
                    
                    try:
                        # Write data and flush/close stdin immediately
                        # This tells aplaymidi we are done sending, but it will keep playing what it got
                        proc.stdin.write(midi_data)
                        proc.stdin.close()
                    except Exception:
                        pass 

                    # ‼️ Sleep for duration minus overlap
                    # We start the NEXT process slightly before this one finishes to cover startup latency
                    # ‼️ Increased overlap to 0.15s to smooth transition
                    overlap = 0.15
                    sleep_time = max(0, duration - overlap)
                    
                    if stop_event.wait(sleep_time):
                        break

                except Exception as e:
                    if not stop_event.is_set():
                        print(f"Playback error: {e}")
                finally:
                    playback_queue.task_done()
        
        # ‼️ Do not start the thread immediately; wait until we buffer a couple chunks

        # State tracking to ensure continuity
        current_tempo = 500000 # Default 120bpm
        current_patches = {} # {channel: patch}
        
        current_prompt = mid_np
        # ‼️ Use the argument passed directly, removing the cap that forced 256 for large values
        chunk_size = args.num_events
        chunk_count = 0
        
        try:
            while True:
                if stop_event.is_set(): break

                # Prevent prompt from growing too large (keep last 2048 tokens)
                if current_prompt.shape[1] > 2048:
                      current_prompt = current_prompt[:, -2048:]
                
                gen_len = current_prompt.shape[1] + chunk_size
                
                print(f"🎹 Generating Chunk {chunk_count+1}...", end="\r")
                sys.stdout.flush()
                
                output = model.generate(
                    prompt=current_prompt, 
                    batch_size=1, 
                    max_len=gen_len,
                    temp=args.temp, 
                    top_p=args.top_p, 
                    top_k=args.top_k,
                    disable_patch_change=disable_patch_change,
                    disable_channels=disable_channels,
                    stop_events=chunk_size, 
                    generator=generator
                )
                
                # Extract *new* tokens for this chunk
                new_tokens = output[0, current_prompt.shape[1]:].tolist()
                
                if len(new_tokens) == 0:
                    print("\n⚠️ Model stopped generating. Reseeding...")
                    continue

                # ‼️ Shave off the end of the previous segment so it doesn't sound like it's ending ‼️
                n_trim = 128 # ‼️ Aggressively remove last 128 events (Increased from 64)
                trimmed_tokens = trim_tokens(tokenizer, new_tokens, n_events_to_trim=n_trim)
                
                # ‼️ DEBUG: Verify trimming (Made unconditional for debugging)
                print(f"\n✂️ Trimmed: {len(new_tokens)} -> {len(trimmed_tokens)} events")

                if len(trimmed_tokens) == 0:
                     print("\n⚠️ Generation shorter than trim amount. Skipping...")
                     continue

                # Detokenize to get the Score using TRIMMED tokens
                chunk_score = tokenizer.detokenize(trimmed_tokens)
                

                if len(chunk_score) > 1:
                    # Track 1 usually exists. Inject metadata at t=0
                    chunk_score[1].insert(0, ['set_tempo', 0, current_tempo])
                    for ch, patch in current_patches.items():
                        chunk_score[1].insert(0, ['patch_change', 0, ch, patch])
                
                # Convert to MIDI bytes
                midi_bytes = MIDI.score2midi(chunk_score)
                
                # ‼️ Calculate duration for smooth playback
                chunk_duration = calculate_chunk_duration(chunk_score, current_tempo)

                # Update State from this chunk (for the NEXT chunk)
                for track in chunk_score[1:]:
                    for event in track:
                        if event[0] == 'set_tempo':
                            current_tempo = event[2]
                        elif event[0] == 'patch_change':
                            current_patches[event[2]] = event[3]

                # Enqueue for playback
                if not stop_event.is_set():
                    # ‼️ Push tuple (data, duration)
                    playback_queue.put((midi_bytes, chunk_duration))
                
                # ‼️ Start player thread only after we have buffered 4 chunks to avoid gaps
                if player_thread is None and playback_queue.qsize() >= 4:
                      print("\n▶️ Buffer filled. Starting Playback...")
                      player_thread = threading.Thread(target=player_worker, daemon=True)
                      player_thread.start()
                
                # Update prompt for next iteration
                # ‼️ Crucial: We update prompt using the TRIMMED output so the model regenerates the trimmed tail
                # output shape is (1, seq_len). We slice it to match prompt + trimmed_new_tokens
                kept_len = current_prompt.shape[1] + len(trimmed_tokens)
                current_prompt = output[:, :kept_len]
                
                chunk_count += 1
        except KeyboardInterrupt:
             print("\n‼️ Ctrl+C detected. Stopping playback...")
             stop_event.set()
             with process_lock:
                 for p in active_processes:
                     try:
                         p.terminate()
                     except: pass
             sys.exit(0)
            
        return


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
    
    # Handle output filename
    base_output = args.output
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
            
        # Unique filename per batch
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
