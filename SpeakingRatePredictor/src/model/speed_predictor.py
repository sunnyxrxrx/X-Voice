from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import Literal

from x_transformers.x_transformers import RotaryEmbedding
from .utils import (
    default,
    exists,
    lens_to_mask,
)
from .modules import (
    MelSpec,
    ConvPositionEmbedding,
    SpeedPredictorLayer,
    GaussianCrossEntropyLoss,
)

class SpeedTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        dropout=0.1,
        ff_mult=4,
        qk_norm=None,
        pe_attn_head=None,
        mel_dim=100,
        num_classes=32,
    ):
        super().__init__()
        self.dim_head = dim // heads
        self.num_classes = num_classes
        self.mel_proj = nn.Linear(mel_dim, dim)
        self.conv_layer = ConvPositionEmbedding(dim=dim)
        self.rotary_embed = RotaryEmbedding(self.dim_head)
        self.transformer_blocks = nn.ModuleList([
                SpeedPredictorLayer(
                    dim=dim,
                    heads=heads,
                    dim_head = self.dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head
                ) for _ in range(depth)
            ])
        self.pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),  # nn.ReLU()
            nn.Linear(dim, num_classes)
        )
        # self.initialize_weights()
        
    # def initialize_weights(self):
        
    def forward(self, x, lens):                     # x.shape = [b, seq_len, d_mel]
        seq_len = x.shape[1]
        mask = lens_to_mask(lens, length=seq_len)   # shape = [b, seq_len]
        
        x = self.mel_proj(x)                        # shape = [b, seq_len, h]
        x = self.conv_layer(x, mask)                # shape = [b, seq_len, h]
        
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope=rope)      # shape = [b, seq_len, h]
            
        # sequence pooling
        weights = self.pool(x)                      # shape = [b, seq_len, 1]
        # 将 padding 位置的 weights 设为 -inf
        weights.masked_fill_(~mask.unsqueeze(-1), -torch.finfo(weights.dtype).max)
        weights = F.softmax(weights, dim=1)         # shape = [b, seq_len, 1]
        x = (x * weights).sum(dim=1)                # shape = [b, h]
        
        output = self.classifier(x)                 # shape: [b, num_classes] 
        return output
    
class SpeedMapper:
    def __init__(
        self, 
        num_classes: Literal[32, 72],
        delta: float = 0.25
    ):
        self.num_classes = num_classes
        self.delta = delta
        
        self.max_speed = float(num_classes) * delta
            
        self.speed_values = torch.arange(0.25, self.max_speed + self.delta, self.delta)
        assert len(self.speed_values) == num_classes, f"Generated {len(self.speed_values)} classes, expected {num_classes}"
    
    def label_to_speed(self, label: torch.Tensor) -> torch.Tensor:
        return self.speed_values.to(label.device)[label] # label * 0.25 + 0.25
    
class SpeedPredictor(nn.Module):
    def __init__(
        self,
        mel_spec_kwargs: dict = dict(),
        loss_type: Literal["GCE", "CE"] = "CE",
        arch_kwargs: dict | None = None,
        sigma_factor: int = 2,
        mel_spec_module: nn.Module | None = None,
        num_channels: int = 100,
        silence_prob: float = 0.0,
        silence_ratio_min: float = 0.2,
        silence_ratio_max: float = 0.8,
    ):
        super().__init__()
        self.num_classes = 32
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels
        self.silence_prob = silence_prob
        self.silence_ratio_min = silence_ratio_min
        self.silence_ratio_max = silence_ratio_max
        self.speed_transformer = SpeedTransformer(**arch_kwargs, num_classes=self.num_classes)
        if loss_type == "GCE":
            self.loss = GaussianCrossEntropyLoss(num_classes=self.num_classes, sigma_factor=sigma_factor)
        elif loss_type == "CE":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.speed_mapper = SpeedMapper(self.num_classes)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def predict_speed(self, audio: torch.Tensor, lens: torch.Tensor | None = None):
        # raw wave
        if audio.ndim == 2:
            audio = self.mel_spec(audio).permute(0, 2, 1)

        batch, seq_len, device = *audio.shape[:2], audio.device
        
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)

        logits = self.speed_transformer(audio, lens)
        probs = F.softmax(logits, dim=-1)
        
        pred_class = torch.argmax(probs, dim=-1)
        pred_speed = self.speed_mapper.label_to_speed(pred_class)
        return pred_speed

    def _inject_silence_mel_batch(
        self,
        inp: float["b n d"],  # noqa: F722
        lens: int["b"],  # noqa: F821
    ):
        # Batch-level gating: either all samples are augmented or none.
        if torch.rand(1, device=inp.device).item() < (1.0 - self.silence_prob):
            return inp, lens

        sil_ratio = torch.empty(1, device=inp.device).uniform_(
            self.silence_ratio_min, self.silence_ratio_max
        ).item()
        sil_len = int(int(lens.max().item()) * sil_ratio)
        if sil_len == 0:
            return inp, lens

        aug_sequences = []
        aug_lens = []
        for sample, sample_len in zip(inp, lens):
            valid_len = int(sample_len.item())
            valid_part = sample[:valid_len]
            front_len = torch.randint(0, sil_len + 1, (1,), device=inp.device).item()
            back_len = sil_len - front_len
            front_silence = inp.new_zeros((front_len, inp.shape[-1]))
            back_silence = inp.new_zeros((back_len, inp.shape[-1]))
            aug_sample = torch.cat((front_silence, valid_part, back_silence), dim=0)

            aug_sequences.append(aug_sample)
            aug_lens.append(valid_len + sil_len)

        padded_aug = pad_sequence(aug_sequences, batch_first=True, padding_value=0.0)
        lens_aug = torch.tensor(aug_lens, device=lens.device, dtype=lens.dtype)
        return padded_aug, lens_aug

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        speed: float["b"],      # speed groundtruth
        lens: int["b"] | None = None,  # noqa: F821
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, device = *inp.shape[:2], inp.device
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        else:
            lens = lens.to(device=device, dtype=torch.long)

        if self.training and self.silence_prob > 0:
            inp, lens = self._inject_silence_mel_batch(inp, lens)

        pred = self.speed_transformer(inp, lens)
        if isinstance(self.loss, GaussianCrossEntropyLoss):
            loss = self.loss(pred, speed, self.device)  # Pass device for GCE
        else:
            loss = self.loss(pred, speed)
        
        return loss
