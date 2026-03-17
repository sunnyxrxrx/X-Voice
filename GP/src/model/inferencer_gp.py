from __future__ import annotations

import os
import torch
import threading
from datetime import timedelta
import json
from random import choice
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn_gp_inference
from f5_tts.model.utils import default, exists, create_derangement, convert_char_to_pinyin, trim_text, available_texts

class Inferencer_gp:
    def __init__(
        self,
        model: CFM,
        checkpoint_path: str,
        root_path: str,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        accelerate_kwargs: dict = dict(),
        mel_spec_type: str = "vocos", 
        frac_lengths_mask: tuple = (0.1, 0.4),
        tokenizer: str = "char",
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0
    ):
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=43200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
            **accelerate_kwargs,
        )

        self.checkpoint_path = checkpoint_path
        self.root_path = root_path
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.mel_spec_type = mel_spec_type
        self.frac_lengths_mask = frac_lengths_mask
        self.tokenizer = tokenizer
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef

        dtype = torch.float32 if self.mel_spec_type == "bigvgan" else None
        from f5_tts.infer.utils_infer import load_checkpoint
        model = load_checkpoint(model, checkpoint_path, str(self.accelerator.device), dtype=dtype, use_ema=True)
        self.model = self.accelerator.prepare(model)
        self.model.eval()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def inference(self, dataset: Dataset, num_workers=16, resumable_with_seed: int = 666):
        generator = torch.Generator()
        if exists(resumable_with_seed):
            generator.manual_seed(resumable_with_seed)

        if self.batch_size_type == "sample":
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn_gp_inference,
                num_workers=num_workers,
                pin_memory=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
                drop_last=False,
            )
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn_gp_inference,
                num_workers=num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be 'sample' or 'frame'")

        dataloader = self.accelerator.prepare(dataloader)

        progress_bar = tqdm(
            range(len(dataloader)),
            desc="Inferencing",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in dataloader:
            text_inputs = batch["text"]
            mel_spec = batch["mel"].permute(0, 2, 1)
            mel_lengths = batch["mel_lengths"]
            rel_paths = batch["rel_paths"]

            self.process_batch(
                mel_spec, 
                text=text_inputs, 
                lens=mel_lengths, 
                rel_paths=rel_paths
            )
            
            progress_bar.update(1)

        self.accelerator.wait_for_everyone()
        print("Inference finished.")

    def process_batch(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722 
        lens: int["b"] | None = None,  # noqa: F821
        rel_paths: str["b nt"] |list[str] = None, 
    ):
        all_files_exist = True
        for i in range(len(rel_paths)):
            pt_file_path = os.path.join(self.root_path, f"{rel_paths[i]}.pt")
            json_file_path = os.path.join(self.root_path, f"{rel_paths[i]}.json")
            
            if not (os.path.exists(pt_file_path) and os.path.exists(json_file_path)):
                all_files_exist = False
                break
        
        if all_files_exist:
            return

        batch, seq_len, device = *inp.shape[:2], self.accelerator.device

        if not exists(lens): # skip
            lens = torch.full((batch,), seq_len, device=device)

        # 为每个text设置一个裁剪比例，范围由 self.frac_lengths_mask = (0.1, 0.4) 决定。
        frac_lengths = torch.zeros((batch,), device=self.accelerator.device).float().uniform_(*self.frac_lengths_mask)

        # TODO
        # random_text = ?
        # text_changed = [random_text]
        
        text_source_input = []
        durations = []
        for i in range(batch):
            original = text[i]
            changed = text_changed[i]
            if self.tokenizer == "pinyin":
                text_source_input.append(convert_char_to_pinyin([changed + " " + original])[0])
            else:
                text_source_input.append(changed + " " + original)
            ref_text_len = len(original.encode("utf-8"))
            gen_text_len = len(changed.encode("utf-8"))
            ref_mel_len = lens[i].item()
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len)
            total_mel_len = min(total_mel_len, 4080)
            durations.append(total_mel_len)

        duration = torch.LongTensor(durations).to(device)
        
        with torch.inference_mode():
            # TODO
            # generated, _ = self.accelerator.unwrap_model(self.model).sample_reverse(
            #     cond=inp,
            #     text=text_source_input,
            #     duration=duration,
            #     lens=lens,
            #     steps=self.nfe_step,
            #     cfg_strength=self.cfg_strength,
            #     sway_sampling_coef=self.sway_sampling_coef,
            # )
            generated = generated.to(torch.float32)
            generated_cpu = generated.cpu()
            for i in range(batch):
                curr_total_len = duration[i].item()
                curr_ref_len = lens[i].item()
                gen_len = curr_total_len - curr_ref_len
                gen_part = generated_cpu[i, :gen_len, :]
                gen_text_content = text_changed[i]

                pt_file_name = f"{rel_paths[i]}.pt"
                json_file_name = f"{rel_paths[i]}.json"
                
                pt_save_path = os.path.join(self.root_path, pt_file_name)
                json_save_path = os.path.join(self.root_path, json_file_name)

                self.save_async(gen_part.clone(), pt_save_path, json_save_path, gen_len, gen_text_content) 

    @staticmethod
    def save_async(tensor_data, pt_path, json_path, gen_len_val, text_content):
        folder = os.path.dirname(pt_path)
        os.makedirs(folder, exist_ok=True)
        torch.save(tensor_data, pt_path)

        metadata = {
            "gen_len": gen_len_val,
            "text": text_content
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)