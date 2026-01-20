import os
import sys
import glob
import random
import subprocess
import argparse


try:
    import audioop
except ImportError:
    try:
        import audioop_lts as audioop
        sys.modules["audioop"] = audioop
    except ImportError:
        print("Error: Python 3.13+ requires 'audioop-lts'. Run: pip install audioop-lts")
        sys.exit(1)

# ‼️ Made SF2 parser import conditional to allow stock mode without dependencies
sf2_parser_available = False
try:
    from sf2utils.sf2parse import Sf2File
    sf2_parser_available = True
except ImportError:
    pass 

# CONFIGURATION
SOUNDFONT_DIR = "/usr/share/soundfonts"
TEMP_CFG = "/tmp/timidity_random.cfg"
OUTPUT_DIR = "output"

def get_random_sf2(directory):
    """Finds a random .sf2 file in the system directory."""
    sf2_files = glob.glob(os.path.join(directory, "**/*.sf2"), recursive=True)
    if not sf2_files:
        print(f"Error: No SoundFonts found in {directory}.")
        sys.exit(1)
    
    selected = random.choice(sf2_files)
    print(f"🎲 Selected SoundFont: {os.path.basename(selected)}")
    return selected

def get_latest_midi(directory):
    """Finds the most recently created MIDI file in the specified directory."""
    if not os.path.exists(directory):
        return None
    
    # Search for common MIDI extensions
    files = []
    for ext in ["*.mid", "*.midi"]:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    if not files:
        return None
    
    # Sort files by creation time (newest first)
    files.sort(key=os.path.getctime, reverse=True)
    return files[0]


def get_random_midi(directory):
    """Finds a random MIDI file in the specified directory."""
    if not os.path.exists(directory):
        return None
    
    # Search for common MIDI extensions
    files = []
    for ext in ["*.mid", "*.midi"]:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    if not files:
        return None
    
    return random.choice(files)

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


    # Providing the full path here causes /path/to/dir//path/to/dir/file.sf2 errors.
    sf2_name = os.path.basename(sf2_path)

    # FIX: Map Melodic instruments using 'bank 0'
    cfg_lines.append("bank 0")
    for midi_program in range(128):
        target_bank, target_preset, target_name = random.choice(valid_mappings)

        line = f"{midi_program} %font \"{sf2_name}\" {target_bank} {target_preset}"
        cfg_lines.append(line)

    # FIX: Use 'drumset 0' instead of 'bank 128' to fix the "Tone bank must be between 0 and 127" error.
    cfg_lines.append("drumset 0")
    for midi_program in range(128):
        target_bank, target_preset, target_name = random.choice(valid_mappings)

        line = f"{midi_program} %font \"{sf2_name}\" {target_bank} {target_preset}"
        cfg_lines.append(line)

    with open(cfg_path, "w") as f:
        f.write("\n".join(cfg_lines))
    
    print(f"   -> Random map written to {cfg_path}")

def main():

    parser = argparse.ArgumentParser(description="Randomize SoundFont and play MIDI with Timidity")

    parser.add_argument("midi", nargs="?", help="Path to the MIDI file. If omitted, plays latest from output/")
    parser.add_argument("-b", "--bpm", type=int, help="Target BPM (percentage based on 120 default)")

    parser.add_argument("-r", "--random", action="store_true", help="Play a random MIDI file from output/ instead of the latest")
    # ‼️ Added --stock argument
    parser.add_argument("-s", "--stock", action="store_true", help="Use stock Timidity configuration (no shuffling)")
    args = parser.parse_args()


    midi_file = args.midi
    if not midi_file:

        if args.random:
            print(f"🔍 No MIDI file specified. Searching for RANDOM in '{OUTPUT_DIR}'...")
            midi_file = get_random_midi(OUTPUT_DIR)
            if not midi_file:
                print(f"Error: No MIDI files found in '{OUTPUT_DIR}' folder.")
                sys.exit(1)
            print(f"✨ Found random: {os.path.basename(midi_file)}")
        else:
            print(f"🔍 No MIDI file specified. Searching for latest in '{OUTPUT_DIR}'...")
            midi_file = get_latest_midi(OUTPUT_DIR)
            if not midi_file:
                print(f"Error: No MIDI files found in '{OUTPUT_DIR}' folder.")
                sys.exit(1)
            print(f"✨ Found newest: {os.path.basename(midi_file)}")

    if not os.path.exists(midi_file):
        print(f"Error: MIDI file not found: {midi_file}")
        sys.exit(1)

    # ‼️ Logic branch: Stock vs Chaos
    if args.stock:
        print("‼️ Using Stock Timidity Configuration")
        cmd = ["timidity", "-id", midi_file]
    else:
        # Check dependency here if we are actually using it
        if not sf2_parser_available:
            print("Error: Please run 'pip install sf2utils' to use random soundfonts, or use --stock.")
            sys.exit(1)

        # 1. Select and Map
        sf2_file = get_random_sf2(SOUNDFONT_DIR)
        generate_chaos_config(sf2_file, TEMP_CFG)

        # 2. Build Command
        cmd = ["timidity", "-id", "-c", TEMP_CFG, midi_file]
    
    if args.bpm:

        # We can't know the original BPM, so we assume 120 is "Standard".
        tempo_percent = int((args.bpm / 120) * 100)
        print(f"   -> Adjusting tempo scale: {tempo_percent}% (Assuming ~120 source)")
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
