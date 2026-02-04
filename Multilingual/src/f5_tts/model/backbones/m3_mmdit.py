from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    M3MMDiTBlock,M3DiTBlock, 
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

class TextEmbedding(nn.Module):
    def __init__(self, out_dim, text_num_embeds, mask_padding=False,average_upsampling=False, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds, out_dim)  # will use 0 as filler token
        #print(f"text_num_embeds:{text_num_embeds}, out dim {out_dim}")
        # 在M3中这两项应该都被置为flase
        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling
        if mask_padding or average_upsampling:
            print("Warning: in M3TTS, mask padding or average upsamping should be false. ")
        self.precompute_max_pos = 8192  # ~88s of 24khz audio
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)
        if conv_layers > 0:
            self.extra_modeling = True
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(out_dim, out_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False
    def average_upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch, text_len, text_dim = text.shape

        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype=torch.bool)
        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim=1)  # [batch]
        valid_lens = valid_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            audio_len = audio_lens[i].item()
            valid_len = valid_lens[i].item()

            if valid_len == 0:
                continue

            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [valid_len, text_dim]

            base_repeat = audio_len // valid_len
            remainder = audio_len % valid_len

            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], drop_text=False, audio_mask: bool["b n"] | None = None) -> int["b nt d"]:
        text_mask = None
        if self.mask_padding:
            text_mask = text == 0
        if drop_text:  # cfg for text
            text = torch.zeros_like(text)
        #print(f"text shape {text.shape}")
        #print(text)
        text = self.text_embed(text)  # b nt -> b nt d
        batch, seq_len, _ = text.shape
        if seq_len > self.freqs_cis.shape[0]:
            # 动态计算当前长度的 pos emb
            current_freqs = precompute_freqs_cis(self.out_dim, seq_len).to(text.device)
        else:
            # 使用缓存
            current_freqs = self.freqs_cis[:seq_len]
        text = text + current_freqs.unsqueeze(0)
        
        if self.extra_modeling:
            if self.mask_padding and text_mask is not None:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask, audio_mask)

        return text


# noised input & masked cond audio embedding


class AudioEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        #print(in_dim,out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"]):
        #print(x.shape)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using MM-DiT blocks


class M3_MMDiT(nn.Module):
    def __init__(
        self,
        *,
        dim, # 音频流、文本流的维度
        joint_depth=8, # joint-dit的深度
        single_depth=8, # single-dit的深度
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100, # 输入音频维度
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(dim, 256)# , mask_padding=text_mask_padding)
        self.text_cond, self.text_uncond = None, None  # text cache
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.joint_depth = joint_depth
        self.single_depth = single_depth
        # Joint-DiT
        self.joint_blocks = nn.ModuleList([])
        for i in range(joint_depth):
            is_last=(i==joint_depth-1)
            self.joint_blocks.append(
                M3MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    context_dim=dim,
                    context_pre_only=is_last,
                    qk_norm=qk_norm,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
            )
        # Single-DiT
        self.single_blocks=nn.ModuleList([
            M3DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
            )for _ in range(single_depth)
        ])
        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in MMDiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.single_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)


        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ):
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x)
        cond = self.audio_embed(cond)
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        return x, c, cond

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        language_ids: list[str] = None, #TODO. 在M3也接入language id
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        text = text + 1 
        text_mask = (text != 0)
        # t: conditioning (time), c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, c_cond, cond_emb = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond, uncond_emb = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            ## cond 如何处理？
            cond = torch.cat((cond_emb,uncond_emb),dim=0)
            text_mask = torch.cat([text_mask, text_mask], dim=0)
        else:
            x, c, cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache
            )
        ## x相当于是音频流，c相当于是文本流，t[b,d], t_broadcast[b,n,d]
        t_broadcast = t.unsqueeze(1).expand_as(cond)
        cf = cond + t_broadcast
        seq_len = x.shape[1]
        text_len = text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)
        for block in self.joint_blocks:
            c,x = block(
                x=x,
                c=c,
                t_c=t,
                t_x=cf,
                mask=mask, 
                context_mask=text_mask,
                rope=rope_audio, c_rope=rope_text
            )
        h=x
        for block in self.single_blocks:
            h = block(
                x=h,
                t=cf,
                mask=mask, rope=rope_audio
            )
        h_final = self.norm_out(h, t)
        output = self.proj_out(h_final)

        return output
