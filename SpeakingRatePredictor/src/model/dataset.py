import os
import json
from importlib.resources import files
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

import pyphen
import pypinyin
import re
from pathlib import Path

DATA_DIR = (Path(__file__) / "../../../data").resolve()

from .modules import MelSpec
from .utils import default

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )

class CustomDataset_old(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        speed_type="syllables",
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.speed_type = speed_type
        num_classes_map = {
            "phonemes": 72,
            "syllables": 32,
            "words": 32
        }
        self.max_label = num_classes_map[self.speed_type]

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            speed = max(row[f"speed_{self.speed_type}"], 0.25)
            speed_class = min(int(speed/0.25 - 1), self.max_label - 1)

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
            "speed": speed,
            "duration": duration,
            "speed_class": speed_class,
        }

class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        speed_type="syllables",
        split="train",
        silence_prob=0.6
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.speed_type = speed_type
        self.split = split
        self.silence_prob = silence_prob
        num_classes_map = {
            "phonemes": 72,
            "syllables": 32,
            "words": 32
        }
        self.max_label = num_classes_map[self.speed_type]

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            speed = max(row[f"speed_{self.speed_type}"], 0.25)
            speed_class = min(int(speed/0.25 - 1), self.max_label - 1)

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            if self.split == "train":
                # Random probability for augmentation case
                prob = torch.rand(1).item()
                silence_threshold = 1.0 - self.silence_prob # (Default) 1 - 0.6 = 0.4
                
                # (Default) 40% probability: Do nothing (0.0 <= prob < 0.4)
                if prob >= silence_threshold:
                    original_len = audio.shape[-1]
                    # Random silence length between 20% and 80% of original length
                    sil_ratio = torch.empty(1).uniform_(0.2, 0.8).item()
                    total_sil_len = int(original_len * sil_ratio)
                    
                    segment_step = self.silence_prob / 3.0              # (Default) 0.2
                    limit_front = silence_threshold + segment_step      # (Default) 0.6
                    limit_back = silence_threshold + (2 * segment_step) # (Default) 0.8                
                    
                    if silence_threshold <= prob < limit_front:
                        # (Default) 20%: Only Front
                        silence = torch.zeros((audio.shape[0], total_sil_len), device=audio.device)
                        audio = torch.cat((silence, audio), dim=-1)
                        
                    elif limit_front <= prob < limit_back:
                        # (Default) 20%: Only Back
                        silence = torch.zeros((audio.shape[0], total_sil_len), device=audio.device)
                        audio = torch.cat((audio, silence), dim=-1)
                        
                    else:
                        # (Default) 20%: Both Front and Back (0.8 <= prob <= 1.0)
                        sil_front = torch.zeros((audio.shape[0], total_sil_len), device=audio.device)
                        sil_back = torch.zeros((audio.shape[0], total_sil_len), device=audio.device)
                        audio = torch.cat((sil_front, audio, sil_back), dim=-1)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
            "speed": speed,
            "duration": duration,
            "speed_class": speed_class,
        }

def load_dataset(
    dataset_name: str,
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    speed_type: Literal["phonemes", "syllables", "words"] = "syllables",
    split: str = "train",
    silence_prob: float = 0.6
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print(f"Loading {split} dataset ...")
    
    suffix = "_val" if split == "val" else ""

    if dataset_type == "CustomDataset":
        rel_data_path = str(DATA_DIR / f"{dataset_name}_speed")
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw{suffix}")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw{suffix}.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel{suffix}.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration{suffix}.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            speed_type=speed_type,
            split=split,
            silence_prob=silence_prob,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, split=split, silence_prob=silence_prob, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(DATA_DIR)),
        )
    print(f"\nSucceed.")
    return train_dataset

def count(text):
    dic = pyphen.Pyphen(lang='en_US')
    total_syllables = 0
    pattern = re.compile(r"[a-zA-Z']+|[\u4e00-\u9fff]")
    tokens = pattern.findall(text)
    
    for token in tokens:
        if '\u4e00' <= token <= '\u9fff':
            total_syllables += 1
        else:
            try:
                syllables = dic.inserted(token.lower()).split("-")
                total_syllables += len(syllables)
            except Exception:
                total_syllables += 1
                
    return total_syllables

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False, drop_last: bool = True
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)

def collate_fn_sp(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)
    
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    speeds = torch.tensor([item["speed"] for item in batch], dtype=torch.float)
    speed_classes = torch.tensor([item["speed_class"] for item in batch], dtype=torch.long)
    durations = [item["duration"] for item in batch]

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        speed=speeds,
        speed_class=speed_classes,
        durations=durations,
        text=text,
        text_lengths=text_lengths,
    )