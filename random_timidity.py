import os
import sys
import glob
import random
import subprocess
import argparse

# ‼️ FIX: Handle the missing 'audioop' module in Python 3.13+ (Common on CachyOS/Arch)
try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
        sys.modules["audioop"] = audioop
    except ImportError:
        print("Error: Python 3.13+ requires 'audioop-lts'. Run: pip install audioop-lts")
        sys.exit(1)

# Import the SF2 parser after the audioop fix
try:
    from sf2utils.sf2parse import Sf2File
except ImportError:
    print("Error: Please run 'pip install sf2utils'")
    sys.exit(1)

# CONFIGURATION
SOUNDFONT_DIR = "/usr/share/soundfonts"
TEMP_CFG = "/tmp/timidity_random.cfg"

def get_random_sf2(directory):
    """Finds a random .sf2 file in the system directory."""
    sf2_files = glob.glob(os.path.join(directory, "**/*.sf2"), recursive=True)
    if not sf2_files:
        print(f"Error: No SoundFonts found in {directory}.")
        sys.exit(1)
    
    selected = random.choice(sf2_files)
    print(f"🎲 Selected SoundFont: {os.path.basename(selected)}")
    return selected

def generate_chaos_config(sf2_path, cfg_path):
    """Parses the SF2 and creates a randomized Timidity mapping."""
    print("   -> Parsing SoundFont...")
    try:
        with open(sf2_path, 'rb') as f:
            sf2 = Sf2File(f)
    except Exception as e:
        print(f"   -> Failed to read SF2 binary: {e}")
        sys.exit(1)

    valid_mappings = []

    # LOGIC: Attempt to get Presets first (The standard mapping)
    for p in sf2.presets:
        name = getattr(p, 'name', '').strip()
        if name.lower() != 'eop' and name != '':
            bank = getattr(p.header, 'bank', 0) if hasattr(p, 'header') else 0
            preset_idx = getattr(p.header, 'preset', 0) if hasattr(p, 'header') else 0
            valid_mappings.append((bank, preset_idx, name))

    # FALLBACK: If Presets are empty, use the Instrument list
    if not valid_mappings:
        print("      (Presets empty, falling back to Instruments list)")
        for i, inst in enumerate(sf2.instruments):
            name = getattr(inst, 'name', '').strip()
            if name.lower() != 'eop' and name != '':
                valid_mappings.append((0, i, name))

    if not valid_mappings:
        print(f"   -> ❌ Fatal: No usable audio data found in {os.path.basename(sf2_path)}")
        sys.exit(1)

    print(f"   -> Found {len(valid_mappings)} usable sounds.")
    
    cfg_lines = [f"dir {os.path.dirname(sf2_path)}"]

    # FIX: Map Melodic instruments using 'bank 0'
    cfg_lines.append("bank 0")
    for midi_program in range(128):
        target_bank, target_preset, target_name = random.choice(valid_mappings)
        line = f"{midi_program} %font \"{sf2_path}\" {target_bank} {target_preset}"
        cfg_lines.append(line)

    # FIX: Use 'drumset 0' instead of 'bank 128' to fix the "Tone bank must be between 0 and 127" error.
    cfg_lines.append("drumset 0")
    for midi_program in range(128):
        target_bank, target_preset, target_name = random.choice(valid_mappings)
        line = f"{midi_program} %font \"{sf2_path}\" {target_bank} {target_preset}"
        cfg_lines.append(line)

    with open(cfg_path, "w") as f:
        f.write("\n".join(cfg_lines))
    
    print(f"   -> Random map written to {cfg_path}")

def main():
    # ‼️ CHANGE: Added argparse to handle MIDI file and optional BPM setting
    parser = argparse.ArgumentParser(description="Randomize SoundFont and play MIDI with Timidity")
    parser.add_argument("midi", help="Path to the MIDI file")
    parser.add_argument("-b", "--bpm", type=int, help="Target BPM (percentage based on 120 default)")
    args = parser.parse_args()

    midi_file = args.midi
    if not os.path.exists(midi_file):
        print(f"Error: MIDI file not found: {midi_file}")
        sys.exit(1)

    # 1. Select and Map
    sf2_file = get_random_sf2(SOUNDFONT_DIR)
    generate_chaos_config(sf2_file, TEMP_CFG)

    # 2. Build Command
    # ‼️ CHANGE: Timidity doesn't set absolute BPM easily; it sets tempo percentage.
    # We use -T [percentage]. If user wants 180 BPM and MIDI is 120, we use -T 150.
    # If no BPM is provided, we don't add the flag.
    cmd = ["timidity", "-id", "-c", TEMP_CFG, midi_file]
    
    if args.bpm:
        # Assuming standard MIDI base of 120 BPM for the calculation
        tempo_percent = int((args.bpm / 120) * 100)
        print(f"   -> Adjusting tempo to ~{args.bpm} BPM ({tempo_percent}%)")
        cmd.insert(1, f"-T{tempo_percent}")

    # 3. Play
    print(f"🎹 Playing {os.path.basename(midi_file)}...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except FileNotFoundError:
        print("Error: 'timidity' not found. Please install it.")

if __name__ == "__main__":
    main()
