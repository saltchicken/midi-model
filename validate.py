import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file as safe_load_file

from midi_model import MIDIModel, MIDIModelConfig
# Reuse the Dataset class from train.py logic (reimplemented here for standalone use)
from train import MidiDataset, get_midi_list

def main():
    parser = argparse.ArgumentParser(description="Standalone Validation for MIDI LoRA")
    
    parser.add_argument("--model", type=str, required=True, help="Path to base model .safetensors")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA adapter .safetensors or directory")
    parser.add_argument("--config", type=str, default="auto", help="Model config")
    parser.add_argument("--data", type=str, required=True, help="Path to directory containing MIDI files for validation")
    
    parser.add_argument("--batch-size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--max-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--lora-rank", type=int, default=64, help="Rank used for training (required to init model)")
    parser.add_argument("--lora-alpha", type=int, default=None, help="Alpha used for training")

    args = parser.parse_args()
    
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank * 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Config
    if args.config == "auto":
        config_path = os.path.join(os.path.dirname(args.model), "config.json")
        config = MIDIModelConfig.from_json_file(config_path) if os.path.exists(config_path) else MIDIModelConfig.from_name("tv2o-medium")
    else:
        config = MIDIModelConfig.from_json_file(args.config) if os.path.exists(args.config) else MIDIModelConfig.from_name(args.config)

    # 2. Load Model
    print(f"Loading Base Model: {args.model}")
    if args.model.endswith(".safetensors"):
        base_state_dict = safe_load_file(args.model)
    else:
        ckpt = torch.load(args.model, map_location="cpu")
        base_state_dict = ckpt.get("state_dict", ckpt)

    model = MIDIModel(config=config)
    model.load_state_dict(base_state_dict, strict=False)

    # 3. Load & Merge LoRA
    print(f"Loading LoRA: {args.lora}")
    # Note: We use the same helper method provided in your generate.py logic via the MIDIModel class
    try:
        model = model.load_merge_lora(args.lora)
    except Exception as e:
        print(f"Error merging LoRA: {e}")
        return

    model.to(device)
    model.eval()

    # 4. Prepare Dataset
    print(f"Loading Data from: {args.data}")
    midi_list = get_midi_list(args.data)
    if not midi_list:
        print("No MIDI files found.")
        return

    dataset = MidiDataset(midi_list, config.tokenizer, max_len=args.max_len, aug=False, check_quality=False, rand_start=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

    # 5. Validation Loop
    total_loss = 0
    total_acc = 0
    total_batches = 0

    print(f"Starting validation on {len(dataset)} files...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            
            x = batch[:, :-1].contiguous()
            y = batch[:, 1:].contiguous()
            
            hidden = model.forward(x)
            hidden = hidden.reshape(-1, hidden.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            x_flat = y_flat[:, :-1]
            
            logits = model.forward_token(hidden, x_flat)
            
            # Compute Loss
            loss = F.cross_entropy(
                logits.view(-1, config.tokenizer.vocab_size),
                y_flat.view(-1),
                reduction="mean",
                ignore_index=config.tokenizer.pad_id
            )
            
            # Compute Accuracy
            out = torch.argmax(logits, dim=-1).flatten()
            labels = y_flat.flatten()
            mask = (labels != config.tokenizer.pad_id)
            out = out[mask]
            labels = labels[mask]
            acc = (out == labels).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_acc = total_acc / total_batches

    print("\n" + "="*30)
    print(f"RESULTS for {os.path.basename(args.lora)}")
    print(f"Avg Loss:     {avg_loss:.4f}")
    print(f"Avg Accuracy: {avg_acc:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
