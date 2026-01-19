import argparse
import os
import random
import json
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Trainer
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from peft import LoraConfig, TaskType
from safetensors.torch import save_file as safe_save_file, load_file as safe_load_file
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import MIDI
from midi_model import MIDIModel, MIDIModelConfig, config_name_list
from midi_tokenizer import MIDITokenizerV2

EXTENSION = [".mid", ".midi"]


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


class MidiDataset(Dataset):
    def __init__(self, midi_list, tokenizer: MIDITokenizerV2, max_len=2048, min_file_size=3000,
                 max_file_size=384000,
                 aug=True, check_quality=False, rand_start=True):

        self.tokenizer = tokenizer
        self.midi_list = midi_list
        self.max_len = max_len
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.aug = aug
        self.check_quality = check_quality
        self.rand_start = rand_start

    def __len__(self):
        return len(self.midi_list)

    def load_midi(self, index):
        path = self.midi_list[index]
        try:
            with open(path, 'rb') as f:
                datas = f.read()
            if len(datas) > self.max_file_size:  # large midi file will spend too much time to load
                raise ValueError("file too large")
            elif len(datas) < self.min_file_size:
                raise ValueError("file too small")
            mid = MIDI.midi2score(datas)
            if max([0] + [len(track) for track in mid[1:]]) == 0:
                raise ValueError("empty track")
            mid = self.tokenizer.tokenize(mid)
            if self.check_quality and not self.tokenizer.check_quality(mid)[0]:
                raise ValueError("bad quality")
            if self.aug:
                mid = self.tokenizer.augment(mid)
        except Exception:
            mid = self.load_midi(random.randint(0, self.__len__() - 1))
        return mid

    def __getitem__(self, index):
        mid = self.load_midi(index)
        mid = np.asarray(mid, dtype=np.int16)
        if self.rand_start:
            start_idx = random.randrange(0, max(1, mid.shape[0] - self.max_len))
            start_idx = random.choice([0, start_idx])
        else:
            max_start = max(1, mid.shape[0] - self.max_len)
            start_idx = (index * (max_start // 8)) % max_start
        mid = mid[start_idx: start_idx + self.max_len]
        mid = mid.astype(np.int64)
        mid = torch.from_numpy(mid)
        return mid

    def collate_fn(self, batch):
        max_len = max([len(mid) for mid in batch])
        batch = [F.pad(mid, (0, 0, 0, max_len - mid.shape[0]), mode="constant", value=self.tokenizer.pad_id) for mid in batch]
        batch = torch.stack(batch)
        return batch


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class SaveIntervalPeftCallback(Callback):
    def __init__(self, interval=50):
        self.interval = interval
        self.last_save_step = -1 

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval > 0 and trainer.global_step > 0 and trainer.global_step % self.interval == 0:
            
            if self.last_save_step == trainer.global_step:
                return
            self.last_save_step = trainer.global_step

            if hasattr(pl_module, "_hf_peft_config_loaded") and pl_module._hf_peft_config_loaded:
                # Determine save directory
                if len(trainer.loggers) > 0 and trainer.loggers[0].save_dir:
                    save_dir = trainer.loggers[0].save_dir
                    name = trainer.loggers[0].name
                    version = trainer.loggers[0].version
                    version = version if isinstance(version, str) else f"version_{version}"
                    save_dir = os.path.join(save_dir, str(name), version)
                else:
                    save_dir = trainer.default_root_dir
                
                ckpt_dir = os.path.join(save_dir, f"checkpoint-{trainer.global_step}")
                latest_dir = os.path.join(save_dir, "lora")
                
                if trainer.is_global_zero:
                    try:
                        # 1. Save Numbered Checkpoint
                        print(f"‼️ Saving Interval LoRA to {ckpt_dir}")
                        pl_module.save_peft(ckpt_dir)

                        # 2. Save 'Latest' Checkpoint (for ease of use)
                        pl_module.save_peft(latest_dir)

                        # 3. Generate Examples
                        print(f"‼️ Generating examples for step {trainer.global_step}")
                        pl_module.gen_example(save_dir)

                    except Exception as e:
                        print(f"Error during interval save/generation: {e}")



class TrainMIDIModel(MIDIModel, pl.LightningModule):
    def __init__(self, config: MIDIModelConfig,
                 lr=2e-4, weight_decay=0.01, warmup=1e3, max_step=1e6, sample_seq=False,
                 gen_example_interval=1, example_batch=8):
        super(TrainMIDIModel, self).__init__(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_step = max_step
        self.sample_seq = sample_seq
        self.gen_example_interval = gen_example_interval
        self.example_batch = example_batch
        self.last_save_step = 0


    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']  # no decay for bias and Norm
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay},
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        if self.sample_seq:  # to reduce vram
            rand_idx = [-1] + random.sample(list(range(y.shape[1] - 2)), min(127, (y.shape[1] - 2) // 2))
            hidden = hidden[:, rand_idx]
            y = y[:, rand_idx]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
        )
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)
        return loss

    @rank_zero_only
    def gen_example(self, save_dir):
        base_dir = f"{save_dir}/sample/{self.global_step}"
        if not os.path.exists(base_dir):
            Path(base_dir).mkdir(parents=True)
        midis = self.generate(batch_size=self.example_batch)
        midis = [self.tokenizer.detokenize(midi) for midi in midis]
        imgs = [self.tokenizer.midi2img(midi) for midi in midis]
        for i, (img, midi) in enumerate(zip(imgs, midis)):
            img.save(f"{base_dir}/0_{i}.png")
            with open(f"{base_dir}/0_{i}.mid", 'wb') as f:
                f.write(MIDI.score2midi(midi))

        # Using train_dataset since validation is disabled
        prompt = train_dataset.load_midi(random.randint(0, len(train_dataset) - 1))

        prompt = np.asarray(prompt, dtype=np.int16)
        ori = prompt[:512]
        img = self.tokenizer.midi2img(self.tokenizer.detokenize(ori))
        img.save(f"{base_dir}/1_ori.png")
        prompt = prompt[:256].astype(np.int64)
        midis = self.generate(prompt, batch_size=self.example_batch)
        midis = [self.tokenizer.detokenize(midi) for midi in midis]
        imgs = [self.tokenizer.midi2img(midi) for midi in midis]
        for i, (img, midi) in enumerate(zip(imgs, midis)):
            img.save(f"{base_dir}/1_{i}.png")
            with open(f"{base_dir}/1_{i}.mid", 'wb') as f:
                f.write(MIDI.score2midi(midi))

    @rank_zero_only
    def save_peft(self, save_dir):
        adapter_name = self.active_adapters()[0]
        adapter_config = self.peft_config[adapter_name]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        adapter_config.save_pretrained(save_dir)
        adapter_state_dict = self.get_adapter_state_dict(adapter_name)

        safe_save_file(adapter_state_dict,
                       os.path.join(save_dir, "adapter_model.safetensors"),
                       metadata={"format": "pt"})


def get_midi_list(path):
    all_files = {
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(path)
        for fname in files
    }
    all_midis = sorted(
        fname for fname in all_files if file_ext(fname) in EXTENSION
    )
    return all_midis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--resume", type=str, default="", help="resume training from ckpt")
    parser.add_argument("--ckpt", type=str, default="", help="load ckpt")
    parser.add_argument("--config", type=str, default="tv2o-medium", help="model config name or file")
    parser.add_argument("--task", type=str, default="train", choices=["train", "lora"], help="Full train or lora")

    # dataset args
    parser.add_argument("--data", type=str, default="data", help="dataset path")
    parser.add_argument("--max-len", type=int, default=2048, help="max seq length for training")
    parser.add_argument("--quality", action="store_true", default=False, help="check dataset quality")
    parser.add_argument("--no-aug", action="store_true", default=False, help="disable data augmentation") 

    # training args
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup-step", type=int, default=1e2, help="warmup step")
    parser.add_argument("--max-step", type=int, default=1e6, help="max training step")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clip val")
    parser.add_argument("--sample-seq", action="store_true", default=False, help="sample midi seq to reduce vram")

    parser.add_argument("--gen-example-interval", type=int, default=0, help="DEPRECATED: Now controlled by --save-interval")
    parser.add_argument("--save-interval", type=int, default=50, help="Save LoRA checkpoint AND generate examples every N steps")
    parser.add_argument("--batch-size-train", type=int, default=1, help="batch size for training")
    parser.add_argument("--batch-size-gen-example", type=int, default=8, help="batch size for generate example")
    parser.add_argument("--workers-train", type=int, default=4, help="workers num for training dataloader")
    parser.add_argument("--acc-grad", type=int, default=2, help="gradient accumulation")
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"], help="accelerator")
    parser.add_argument("--precision", type=str, default="bf16-true", help="precision")
    parser.add_argument("--devices", type=int, default=-1, help="devices num")
    parser.add_argument("--nodes", type=int, default=1, help="nodes num")
    parser.add_argument("--disable-benchmark", action="store_true", default=False, help="disable cudnn benchmark")
    parser.add_argument("--log-step", type=int, default=1, help="log training loss every n steps")
    parser.add_argument("--lora-rank", type=int, default=64, help="lora rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="lora alpha")

    parser.add_argument("--name", type=str, default=None, help="experiment name (output directory name)")

    opt = parser.parse_args()


    if opt.name is None:
        data_name = os.path.basename(os.path.normpath(opt.data))
        opt.name = f"{data_name}_{opt.lora_rank}"
        print(f"‼️ No name provided. Auto-generated name: {opt.name}")
    elif opt.name == "default":
        # Handle case where user might explicitly pass "default" if they really wanted the old behavior, or just fallback
        pass

    if opt.lora_alpha is None:
        opt.lora_alpha = opt.lora_rank * 2
        print(f"‼️ --lora-alpha was left at None, so it is being set to {opt.lora_alpha} (rank * 2)")

    print(opt)

    torch.set_float32_matmul_precision('medium')

    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
    if not os.path.exists("sample"):
        os.mkdir("sample")
    pl.seed_everything(opt.seed)
    
    print("---load dataset---")
    if opt.config in config_name_list:
        config = MIDIModelConfig.from_name(opt.config)
    else:
        config = MIDIModelConfig.from_json_file(opt.config)
    tokenizer = config.tokenizer
    midi_list = get_midi_list(opt.data)


    if not midi_list:
        raise ValueError(f"No MIDI files found in {opt.data}")

    random.shuffle(midi_list)
    
    print(f"ℹ️ All {len(midi_list)} files will be used for training (Validation disabled).")
    train_midi_list = midi_list

    train_dataset = MidiDataset(train_midi_list, tokenizer, max_len=opt.max_len, aug=not opt.no_aug, check_quality=opt.quality,
                                    rand_start=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size_train,
        shuffle=True,
        persistent_workers=True,
        num_workers=opt.workers_train,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    devices_count = opt.devices
    if devices_count == -1:
        if opt.accelerator == "cpu":
            devices_count = 1
        elif opt.accelerator == "gpu" or opt.accelerator == "auto":
            devices_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            devices_count = 1
    devices_count = max(1, devices_count)

    effective_batch_size = opt.batch_size_train * devices_count * opt.nodes * opt.acc_grad
    steps_per_epoch = len(train_dataset) / effective_batch_size

    if steps_per_epoch > 0:
        total_epochs = opt.max_step / steps_per_epoch
        print(f"--- Training Duration Estimate ---")
        print(f"  Effective Batch Size: {effective_batch_size}")
        print(f"  Steps per Epoch:      {steps_per_epoch:.2f}")
        print(f"  Max Steps:            {opt.max_step}")
        print(f"  Estimated Epochs:     {total_epochs:.2f}")
        print(f"----------------------------------")


    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    model = TrainMIDIModel(config, lr=opt.lr, weight_decay=opt.weight_decay,
                           warmup=opt.warmup_step, max_step=opt.max_step,
                           sample_seq=opt.sample_seq, gen_example_interval=opt.gen_example_interval,
                           example_batch=opt.batch_size_gen_example)
    if opt.ckpt:
        if opt.ckpt.endswith(".safetensors"):
            state_dict = safe_load_file(opt.ckpt)
        else:
            ckpt = torch.load(opt.ckpt, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    elif opt.task == "lora":
        raise ValueError("--ckpt must be set to train lora")
    if opt.task == "lora":
        model.requires_grad_(False)
        lora_config = LoraConfig(
            r=opt.lora_rank,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            lora_alpha=opt.lora_alpha,
            lora_dropout=0
        )
        model.add_adapter(lora_config)
    
    print("---start train---")
    
    callbacks = []
    

    callbacks.append(SaveIntervalPeftCallback(interval=opt.save_interval))

    logger = TensorBoardLogger("lightning_logs", name=opt.name)

    trainer = Trainer(
        enable_checkpointing=False,
        precision=opt.precision,
        accumulate_grad_batches=opt.acc_grad,
        gradient_clip_val=opt.grad_clip,
        accelerator=opt.accelerator,
        devices=opt.devices,
        num_nodes=opt.nodes,
        max_steps=opt.max_step,
        benchmark=not opt.disable_benchmark,
        log_every_n_steps=1,
        strategy="auto",
        callbacks=callbacks,
        logger=logger,

        num_sanity_val_steps=0,
        limit_val_batches=0.0
    )
    ckpt_path = opt.resume
    if ckpt_path == "":
        ckpt_path = None
    
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
