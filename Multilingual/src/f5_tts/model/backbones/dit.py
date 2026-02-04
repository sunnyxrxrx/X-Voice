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

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, 
        text_num_embeds, 
        text_dim, 
        mask_padding=True, 
        average_upsampling=False, 
        conv_layers=0, 
        conv_mult=2,
        num_languages=None,
        lang_dim=None,
        infill_lang_type=None,
        lang_dropout_prob=0.0,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.infill_lang_type=infill_lang_type
        self.num_languages = num_languages
        print(f"[debug]: Number of languages: {num_languages}. Language dim: {lang_dim}, drop probability: {lang_dropout_prob}")
        print("[debug]: If number of languages is None, language ids will be infilled in time t, instead of per token. If language dim is None, no dropout.")
        self.lang_dropout_prob = lang_dropout_prob
        if self.num_languages is not None:
            if lang_dim is None: 
                lang_dim = text_dim
                # 只是前向支持，之前没有drop的逻辑，所以加载ckpt时维度会不匹配，在此修正
                assert lang_dropout_prob == 0, "If you want to drop language ids, please give lang_dim explicitly."
                self.lang_embed = nn.Embedding(num_languages, lang_dim) 
            else:
                self.lang_embed = nn.Embedding(num_languages + 1, lang_dim) # 加一个维度作为未知语言
            nn.init.normal_(self.lang_embed.weight, std=0.02)
            if self.infill_lang_type =="token_concat":
                self.fusion = nn.Linear(text_dim + lang_dim, text_dim)
                with torch.no_grad():
                    self.fusion.weight.fill_(0.0)
                    # 让前半部分，对应text_dim，成为单位阵
                    self.fusion.weight[:, :text_dim] = torch.eye(text_dim)
                    if self.fusion.bias is not None:
                        self.fusion.bias.fill_(0.0)
            elif self.infill_lang_type == "ada":
                self.lang_ada_layer = nn.Linear(lang_dim, text_dim * 2) 
                # 初始化为 0，确保训练初期模型只看音素
                nn.init.zeros_(self.lang_ada_layer.weight)
                nn.init.zeros_(self.lang_ada_layer.bias)
        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # ~88s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
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

    def forward(self, text: int["b nt"], seq_len, drop_text=False, audio_mask: bool["b n"] | None = None, language_ids=None):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d
        # concat language embedding
        if self.num_languages is not None and language_ids is not None:
            # 只有在训练模式下才执行Dropout
            current_lang_ids = language_ids.clone()
            if self.training:
                # 产生一个掩码，以 lang_dropout_prob的概率将id替换为特殊的index 
                mask = torch.rand(current_lang_ids.shape, device=text.device) < self.lang_dropout_prob
                current_lang_ids[mask] = self.num_languages # 指向预留的未知语种位
            
            l_emb = self.lang_embed(current_lang_ids) # [b, lang_dim]
            if self.infill_lang_type =="token_concat":
                # [b, 1, lang_dim] -> [b, n, lang_dim]
                l_emb = l_emb.unsqueeze(1).expand(-1, text.size(1), -1)
                merged = torch.cat([text, l_emb], dim=-1)
                text = self.fusion(merged) 
            elif self.infill_lang_type == "ada":
                adaln = self.lang_ada_layer(l_emb).unsqueeze(1) 
                scale, shift = adaln.chunk(2, dim=-1)
                text = text * (1 + scale) + shift
    
        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask, audio_mask)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
        audio_mask: bool["b n"] | None = None,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
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
        languages: list | None = None, 
        infill_lang_type: str | None=None, # how to infill languages id
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,   
        use_ctc: bool = False,
        lang_dim: int | None = None, # default to text dim
        lang_dropout_prob=0.0,
    ):
        super().__init__()

        self.infill_lang_type = infill_lang_type
        self.time_embed = TimestepEmbedding(dim)
        self.languages = languages
        if self.languages is not None:
            self.lang_to_id = {lang: i for i, lang in enumerate(self.languages)}
            self.num_languages = len(self.languages)
            if self.infill_lang_type is None or self.infill_lang_type in ["add_only","concat"]: # 如果infill_lang_type为空，默认为"add_only"
                self.lang_embed = nn.Embedding(self.num_languages, dim)
                nn.init.normal_(self.lang_embed.weight, std=0.02)
                if self.infill_lang_type == "concat": 
                    self.cond_fusion = nn.Sequential(
                        nn.Linear(dim * 2, dim),
                        nn.SiLU(),
                        nn.Linear(dim, dim)
                    )
        text_embed_num_langs = self.num_languages if (languages and infill_lang_type in ["token_concat","ada"]) else None
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            num_languages=text_embed_num_langs,
            lang_dim=lang_dim,
            infill_lang_type=infill_lang_type,
            lang_dropout_prob=lang_dropout_prob,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim,use_rmsnorm)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_head = nn.Linear(dim, text_num_embeds + 1)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
        lang_ids_tensor=None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text, audio_mask=audio_mask, language_ids=lang_ids_tensor)
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_lens[i].item(),
                        drop_text=drop_text,
                        audio_mask=audio_mask,
                        language_ids=lang_ids_tensor[i:i+1] if lang_ids_tensor is not None else None,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(text_embed_list, batch_first=True, padding_value=0)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

        return x

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
        language_ids: list[str] = None,
        return_ctc: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if self.languages is not None:
            if language_ids is None:
                raise ValueError("language_ids must be provided for multilingual training.")
            lang_ids_tensor = torch.tensor(
                    [self.lang_to_id[lang] for lang in language_ids], 
                    dtype=torch.long, 
                    device=text.device
                )
            if self.infill_lang_type in ["token_concat","ada"]:
                pass # process in TextEmbedding
            else:
                lang_emb = self.lang_embed(lang_ids_tensor)
                if not self.infill_lang_type or self.infill_lang_type=="add_only":
                    t += lang_emb
                elif self.infill_lang_type=="concat":
                    joint_cond = torch.cat([t, lang_emb], dim=-1)
                    t = self.cond_fusion(joint_cond)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask, lang_ids_tensor=lang_ids_tensor,
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask, lang_ids_tensor=lang_ids_tensor,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask, lang_ids_tensor=lang_ids_tensor,
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        
        if self.use_ctc and return_ctc:
            ctc_logits = self.ctc_head(x)
            return output, ctc_logits
        return output
