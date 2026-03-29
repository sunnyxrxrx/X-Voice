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

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx, list_str_to_idx_ipa, list_list_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
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
        ctc_loss_weight: float = 0.1
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
        print(f"[debug]: {audio_drop_prob=}, {cond_drop_prob=}(text, audio, and language), {lang_drop_prob=}(language).")
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
        prompt_ids: torch.Tensor | None = None,
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

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        if reverse:
            # 原逻辑
            # cond = F.pad(cond, (0, 0, max_duration - cond_seq_len, 0), value=0.0)
            # cond_mask = F.pad(cond_mask, (max_duration - cond_mask.shape[-1], 0), value=False)
            # 修正后
            full_cond = torch.zeros((batch, max_duration, self.num_channels), device=device, dtype=cond.dtype)
            cond_mask = torch.zeros((batch, max_duration), device=device, dtype=torch.bool)
            for b in range(batch):
                curr_ref_len = lens[b] if lens is not None else cond_seq_len
                curr_total_len = duration[b]
                gen_len = curr_total_len - curr_ref_len
                # 把 Reference 填入到最后
                # 形状: [Ref_Len, Dim]
                full_cond[b, gen_len:curr_total_len, :] = cond[b, :curr_ref_len, :]
                
                # Mask: 前面生成的为 False (需要预测)，后面 Ref 为 True (固定)
                # F5-TTS 的 cond_mask 含义: True = 固定/已知, False = 未知/生成
                cond_mask[b, gen_len:curr_total_len] = True
            cond = full_cond
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
            if cfg_schedule == "linear":
                if t > cfg_decay_time:
                    # linear decline
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
                    infer_mode=infer_mode,
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
                layered=layered,
                infer_mode=infer_mode,
                prompt_ids=prompt_ids,
            )
            
            if layered:
                pred, text_pred, null_pred = torch.chunk(pred_cfg, 3, dim=0)
                delta_lang = pred - text_pred
                delta_content = text_pred - null_pred  # 内容增量：从噪音到“平均发音”
                res = null_pred + (1.0 + current_cfg2) * delta_content + (1.0 + current_cfg) * delta_lang
                #res = null_pred + (1.0 + current_cfg) * (pred - text_pred) + (1.0 + current_cfg2) * (text_pred - null_pred)
                if 0.3 < t < 0.6:
                    print(f"content.mean: {delta_content.mean()}, lang.mean: {delta_lang.mean()}") 
            else:
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                res = pred + (pred - null_pred) * current_cfg
            
            # # 缩放res
            # rescale_phi = 0.7             
            # std_pos = torch.std(pred, dim=(1, 2), keepdim=True) + 1e-5
            # std_cfg = torch.std(res, dim=(1, 2), keepdim=True) + 1e-5
            # # 缩放后的 v
            # res_rescaled = res * (std_pos / std_cfg)
            # # 最终的 v 是 原始v 和 缩放v 的平滑混合
            # res= rescale_phi * res_rescaled + (1.0 - rescale_phi) * res
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
        noise_scheduler: str | None = None,
        language_ids: list[str] | torch.Tensor | None, # 在 cross lingual中，传到cfm的就是id数字了
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

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

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
                mask=mask, language_ids=language_ids,
                return_ctc=True,
            )
        else:
            # x_ctx=cond, t, y'=text, x_t=φ
            # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
            pred = self.transformer(
                x=φ, cond=cond, text=text, time=time, 
                drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_lang=drop_lang,
                mask=mask, language_ids=language_ids,
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
