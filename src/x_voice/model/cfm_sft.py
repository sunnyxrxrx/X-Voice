"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from x_voice.model.modules import MelSpec
from x_voice.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx, list_str_to_idx_ipa, list_list_to_idx,
    list_str_to_tensor,
    mask_from_prompt_lens,
    prefix_text_padding,
    build_prefixed_language_ids,
    build_prefixed_language_ids_tokenwise,
)


class CFM_SFT(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        tokenizer = "char", 
        sigma=0.0, # for flow matching
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        lang_drop_prob=0.0,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        ctc_loss_weight: float = 0.1,
        use_total_text: bool = False,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.lang_drop_prob = lang_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        self.tokenizer = tokenizer
        self.ctc_loss_weight = ctc_loss_weight
        self.use_total_text = use_total_text

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str] | list[list[str]],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=8192,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        language_ids:  list[str] | torch.Tensor | None = None, 
        cfg_schedule=None,
        cfg_decay_time=0.0, # for cfg_schedule
        reverse=False,
        layered=False,
        cfg_strength2=0.0, # for layered cfg
        infer_mode=True,
        codeswitch_time_unknown_lang: bool = False,
    ):
        self.eval()
        # raw wave
        # print(f"cfg schedule {cfg_schedule}")
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list) and isinstance(text[0],str):
            if exists(self.vocab_char_map):
                if self.tokenizer.startswith("ipa"):
                    text = list_str_to_idx_ipa(text, self.vocab_char_map, self.tokenizer, language_ids=language_ids).to(device)
                else:
                    text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        elif isinstance(text, list) and isinstance(text[0],list):
            assert exists(self.vocab_char_map)
            text = list_list_to_idx(text, self.vocab_char_map).to(device)
            

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        if not self.use_total_text:
            prefix_token_id = len(self.vocab_char_map)
            anchor_token_ids = list_str_to_idx([". "], self.vocab_char_map).to(device=device, dtype=text.dtype)[0]
            if isinstance(language_ids, list):
                language_ids = torch.tensor(
                    [self.transformer.lang_to_id[l] for l in language_ids],
                    dtype=torch.long,
                    device=device,
                )
            else:
                language_ids = language_ids.to(device=device, dtype=torch.long)
            base_text = text
            text = prefix_text_padding(
                text,
                duration,
                lens,
                prefix_token_id=prefix_token_id,
                anchor_token_ids=anchor_token_ids,
            )
            time_language_ids = None
            if language_ids.dim() == 2:
                language_ids = build_prefixed_language_ids_tokenwise(
                    text=base_text,
                    total_lens=duration,
                    prompt_lens=lens,
                    language_ids=language_ids,
                    anchor_token_ids=anchor_token_ids,
                    unknown_lang_id=self.transformer.num_languages,
                )
            else:
                time_language_ids = language_ids.clone()
                language_ids = build_prefixed_language_ids(
                    text=base_text,
                    total_lens=duration,
                    prompt_lens=lens,
                    language_ids=language_ids,
                    anchor_token_ids=anchor_token_ids,
                    unknown_lang_id=self.transformer.num_languages,
                )

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        if reverse:
            cond = F.pad(cond, (0, 0, max_duration - cond_seq_len, 0), value=0.0)
            cond_mask = F.pad(cond_mask, (max_duration - cond_mask.shape[-1], 0), value=False)
        else:
            cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
            cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        if no_ref_audio:
            cond = torch.zeros_like(cond)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            current_cfg = cfg_strength
            current_cfg2 = cfg_strength2
            if cfg_schedule == "square":
                if t > cfg_decay_time:
                    current_cfg = cfg_strength * ((1 - t) ** 2)
                    current_cfg2 = cfg_strength2 * ((1 - t) ** 2)
            elif cfg_schedule == "cosine":
                if t > cfg_decay_time:
                    # cosine decline
                    normalized_t = (t - cfg_decay_time) / (1.0 - cfg_decay_time)
                    current_cfg = cfg_strength * torch.cos(0.5 * torch.pi * normalized_t)
                    current_cfg2 = cfg_strength2 * torch.cos(0.5 * torch.pi * normalized_t)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    drop_lang=False,
                    cache=True,
                    language_ids=language_ids,
                    time_language_ids=time_language_ids,
                    infer_mode=infer_mode,
                    codeswitch_time_unknown_lang=codeswitch_time_unknown_lang,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
                language_ids=language_ids,
                time_language_ids=time_language_ids,
                layered=layered,
                infer_mode=infer_mode,
                codeswitch_time_unknown_lang=codeswitch_time_unknown_lang,
            )
            
            if layered:
                pred, text_pred, null_pred = torch.chunk(pred_cfg, 3, dim=0)
                delta_audio = pred - text_pred
                delta_content = text_pred - null_pred
                warmup_gate = torch.clamp(t / 0.01, max=1.0)
                res = null_pred + (1.0 + current_cfg2 * warmup_gate) * delta_content + (1.0 + current_cfg) * delta_audio
            else:
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                res = pred + (pred - null_pred) * current_cfg
            
            return res


        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        # y0：[batch_size, max_duration, num_channels]
        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, int(steps + 1), device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        if torch.isnan(out).any():
            print("Detected NaN in generated buffer!")
        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        prompt_lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        language_ids: list[str] | torch.Tensor | None, # Cross-lingual mode passes numeric language ids into CFM.
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                pre_text=text
                if self.tokenizer.startswith("ipa"):
                    text = list_str_to_idx_ipa(text, self.vocab_char_map, self.tokenizer, language_ids=language_ids).to(device)
                else:
                    text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device) # [batch_size,squence_len]
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        
        if not self.use_total_text:
            prefix_token_id = len(self.vocab_char_map)
            anchor_token_ids = list_str_to_idx([". "], self.vocab_char_map).to(device=device, dtype=text.dtype)[0]
            if isinstance(language_ids, list):
                language_ids = torch.tensor(
                    [self.transformer.lang_to_id[l] for l in language_ids],
                    dtype=torch.long,
                    device=device,
                )
            else:
                language_ids = language_ids.to(device=device, dtype=torch.long)
            base_text = text
            text = prefix_text_padding(
                text,
                lens,
                prompt_lens,
                prefix_token_id=prefix_token_id,
                anchor_token_ids=anchor_token_ids,
            )
            time_language_ids = language_ids.clone()
            language_ids = build_prefixed_language_ids(
                text=base_text,
                total_lens=lens,
                prompt_lens=prompt_lens,
                language_ids=language_ids,
                anchor_token_ids=anchor_token_ids,
                unknown_lang_id=self.transformer.num_languages,
            )

        # get a random span to mask out for training conditionally
        rand_span_mask = mask_from_prompt_lens(prompt_lens, lens)

        if exists(mask):
            rand_span_mask &= mask

        # clean mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        rand = random()
        if rand < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
            drop_lang = True
        elif rand < self.cond_drop_prob + self.lang_drop_prob:
            drop_text = False
            drop_lang = True
        else:
            drop_text = False
            drop_lang = False
        
        text_for_ctc = text 
        use_ctc = getattr(self.transformer, "use_ctc", False)
        if use_ctc:
            pred, ctc_logits = self.transformer(
                x=φ, cond=cond, text=text, time=time, 
                drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_lang=drop_lang,
                mask=mask, language_ids=language_ids, time_language_ids=time_language_ids,
                return_ctc=True,
            )
        else:
            # x_ctx=cond, t, y'=text, x_t=φ
            # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
            pred = self.transformer(
                x=φ, cond=cond, text=text, time=time, 
                drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_lang=drop_lang,
                mask=mask, language_ids=language_ids, time_language_ids=time_language_ids,
                return_ctc=False,
            )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        
        if use_ctc and self.ctc_loss_weight > 0 and not drop_text:
            # [B, N, C] -> [N, B, C] and log_softmax
            ctc_input = ctc_logits.transpose(0, 1).log_softmax(dim=2)
            # text token starts from 0, padding is -1
            # CTC blank is 0, so padding becomes 0 when token id plus 1 
            target_lengths = (text_for_ctc != -1).sum(dim=1)
            ctc_target = text_for_ctc + 1 
            input_lengths = lens

            ctc_loss = F.ctc_loss(
                ctc_input, 
                ctc_target, 
                input_lengths, 
                target_lengths, 
                blank=0, 
                zero_infinity=True
            )
            
            loss = loss + self.ctc_loss_weight * ctc_loss

        return loss.mean(), cond, pred
