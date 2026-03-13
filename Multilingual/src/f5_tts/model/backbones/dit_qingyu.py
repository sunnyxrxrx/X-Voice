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
        text_infill_lang_type=None,
        lang_dropout_prob=0.0,
        share_lang_embed=False,
        lang_embed_obj=None,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.text_infill_lang_type = text_infill_lang_type
        self.num_languages = num_languages
        print(f"[debug]: Number of languages: {num_languages}. Language dim: {lang_dim}, drop probability: {lang_dropout_prob}")
        print("[debug]: If number of languages is None, language ids will  be infilled in time t only, instead of per token. If lang_dim is None, no dropout.")
        print("[debug]: In new version, should give lang_dim, or set lang_dim to lang_dim_of_t, share_lang_embed to True and give lang_embed_obj.")
        print("[debug]: In new version, lang_drop_prob has no use, leave it to be handled in cfm.py ")
        # self.lang_dropout_prob = lang_dropout_prob
        if self.num_languages is not None:
            if lang_dim is None and not share_lang_embed: 
                lang_dim = text_dim
                # 只是前向支持，之前没有drop的逻辑，所以加载ckpt时维度会不匹配，在此修正
                assert lang_dropout_prob == 0, "If you want to drop language ids, please give lang_dim explicitly."
                self.lang_embed = nn.Embedding(num_languages, lang_dim) 
                nn.init.normal_(self.lang_embed.weight, std=0.02)
            elif share_lang_embed:
                assert lang_embed_obj is not None
                self.lang_embed = lang_embed_obj
            else:
                self.lang_embed = nn.Embedding(num_languages + 1, lang_dim) # 加一个维度作为未知语言
                nn.init.normal_(self.lang_embed.weight, std=0.02)
            
            if self.text_infill_lang_type == "token_concat":
                self.fusion = nn.Linear(text_dim + lang_dim, text_dim)
                
            elif self.text_infill_lang_type == "ada":
                self.lang_ada_layer = nn.Linear(lang_dim, text_dim * 2) 
        
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

    def average_upsample_text_by_mask(self, text, text_mask, target_lens):
        batch, max_seq_len, text_dim = text.shape
        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = int(text_lens[i].item())
            audio_len = int(target_lens[i].item())

            if text_len == 0 or audio_len <= 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [text_len, text_dim]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False, drop_lang=False, language_ids=None):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        valid_pos_mask = None
        if torch.is_tensor(seq_len):
            seq_len = seq_len.to(device=text.device, dtype=torch.long)
            max_seq_len = int(seq_len.max().item())
        else:
            max_seq_len = int(seq_len)

        text = text[:, :max_seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)

        if torch.is_tensor(seq_len):
            seq_pos = torch.arange(max_seq_len, device=text.device).unsqueeze(0)
            valid_pos_mask = seq_pos < seq_len.unsqueeze(1)
            text = text.masked_fill(~valid_pos_mask, 0)

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # [b nt] -> [b nt d]
        if valid_pos_mask is not None:
            # Keep short-sample tail strictly zero (equivalent to per-sample pad_sequence(..., 0)).
            text = text.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)

        # concat language embedding
        if self.num_languages is not None and language_ids is not None:
            # 统一 language id 的维度
            if language_ids.dim() == 2: # [b, nt]，用于cross lingual
                language_ids = language_ids[:, :max_seq_len]
                # 这里任意pad一个值就好，因为后面concat之后还是要根据text_mask把后面变为0的
                language_ids = F.pad(language_ids, (0, max_seq_len - language_ids.shape[1]), value=0) 
            else:
                assert language_ids.dim() == 1 # [b]，用于正常的单一语言
                # [b] -> [b, 1] -> [b, nt]
                language_ids = language_ids.unsqueeze(1).expand(-1, text.size(1))
                
            current_lang_ids = language_ids.clone()

            # 用于 cfg 或保留文本丢弃语言
            if drop_lang:
                current_lang_ids = torch.full_like(current_lang_ids, self.num_languages)
            
            l_emb = self.lang_embed(current_lang_ids) # [b, nt, lang_dim] 
            assert text.shape[0] == l_emb.shape[0] and text.shape[1] == l_emb.shape[1], f"Shape mismatch: text vs lang_ids"
            if self.text_infill_lang_type == "token_concat":
                merged = torch.cat([text, l_emb], dim=-1)
                text = self.fusion(merged) 
            elif self.text_infill_lang_type == "ada":
                adaln = self.lang_ada_layer(l_emb)
                scale, shift = adaln.chunk(2, dim=-1)
                text = text * (1 + scale) + shift
    
        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb; for variable seq lengths, only add positions within each sample's valid range.
            freqs = self.freqs_cis[:max_seq_len, :]
            if valid_pos_mask is not None:
                freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
            text = text + freqs

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            if torch.is_tensor(seq_len):
                target_lens = seq_len.to(device=text.device, dtype=torch.long)
            else:
                target_lens = torch.full((text.shape[0],), int(seq_len), device=text.device, dtype=torch.long)

            text = self.average_upsample_text_by_mask(text, ~text_mask, target_lens)

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
        text_infill_lang_type: str | None=None, # how to infill languages id
        time_infill_lang_type: str | None=None, # how to infill languages id
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,   
        use_ctc: bool = False,
        lang_dim: int | None = None, # default to text dim
        lang_dim_in_t: int | None = None, # default to dim
        lang_dropout_prob=0.0, # 后续不再使用，仅供前向支持
        drop_lang_in_time: bool | None =False, # 仅供前向支持
        share_lang_embed: bool | None =False,
    ):
        super().__init__()

        
        self.time_embed = TimestepEmbedding(dim)
        self.languages = languages
        self.drop_lang_in_time = drop_lang_in_time
        if self.languages is not None:
            self.time_infill_lang_type = time_infill_lang_type 
            self.text_infill_lang_type = text_infill_lang_type
            self.lang_to_id = {lang: i for i, lang in enumerate(self.languages)}
            self.num_languages = len(self.languages)
            self.lang_dim_in_t = lang_dim_in_t if lang_dim_in_t is not None else dim
            if self.time_infill_lang_type in ["add_only",  "time_concat"]:
                if not self.drop_lang_in_time:
                    print("[debug]: if you want to drop language id in t, please set drop_lang_in_time to True.")
                    self.lang_embed = nn.Embedding(self.num_languages, self.lang_dim_in_t)
                else:
                    self.lang_embed = nn.Embedding(self.num_languages + 1, self.lang_dim_in_t)
                nn.init.normal_(self.lang_embed.weight, std=0.02)
                
                if self.time_infill_lang_type == "time_concat":  
                    self.cond_fusion = nn.Sequential(
                        nn.Linear(dim + self.lang_dim_in_t, dim),
                        nn.SiLU(),
                        nn.Linear(dim, dim)
                    )
                elif self.time_infill_lang_type == "add_only":
                    self.lang_proj = nn.Linear(lang_dim_in_t, dim)
        
        # 要想在文本维度注入language id，必须传入正确的num_langguages
        text_embed_num_langs = self.num_languages if (languages and self.text_infill_lang_type in ["token_concat", "ada"]) else None
        
        if text_dim is None:
            text_dim = mel_dim
        self.text_dim = text_dim
        lang_embed_obj = self.lang_embed if share_lang_embed else None
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            num_languages=text_embed_num_langs,
            lang_dim=lang_dim,
            text_infill_lang_type=text_infill_lang_type,
            lang_dropout_prob=lang_dropout_prob,
            share_lang_embed=share_lang_embed,
            lang_embed_obj=lang_embed_obj,
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

        self.norm_out = AdaLayerNorm_Final(dim, use_rmsnorm)  # final modulation
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
        if self.time_infill_lang_type == "time_concat":
            first_linear = self.cond_fusion[0]   
            with torch.no_grad():
                # 只将后半部分（对应 lang_dim 的列）设为 0
                first_linear.weight[:, self.dim:].fill_(0.0)
        elif self.time_infill_lang_type == "add_only":
            nn.init.constant_(self.lang_proj.weight, 0)
            nn.init.constant_(self.lang_proj.bias, 0)
        if self.text_infill_lang_type == "token_concat":
            with torch.no_grad():
                self.text_embed.fusion.weight.fill_(0.0)
                # 让前半部分，对应text_dim，成为单位阵
                self.text_embed.fusion.weight[:, :self.text_dim] = torch.eye(self.text_dim)
                if self.text_embed.fusion.bias is not None:
                    self.text_embed.fusion.bias.fill_(0.0)
        elif self.text_infill_lang_type == "ada":
            nn.init.constant_(self.text_embed.lang_ada_layer.weight, 0)
            nn.init.constant_(self.text_embed.lang_ada_layer.bias, 0)
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
        drop_lang: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
        lang_ids_tensor: torch.Tensor | None = None, # [b] / [b, nt]
    ):
        if drop_text:
            state = "null"
        elif drop_lang:
            state = "text_only"
        else:
            state = "full"
        if not hasattr(self, 'text_cache'):
            self.text_cache = {}
        
        if not cache or state not in self.text_cache:
        # if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                seq_len = x.shape[1]
            else:
                seq_len = audio_mask.sum(dim=1)  # per-sample valid speech length
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text, drop_lang=drop_lang, language_ids=lang_ids_tensor)
            if cache:
                # if drop_text:
                #     self.text_uncond = text_embed
                # else:
                #     self.text_cond = text_embed
                self.text_cache[state] = text_embed

        if cache:
            # if drop_text:
            #     text_embed = self.text_uncond
            # else:
            #     text_embed = self.text_cond
            text_embed = self.text_cache[state]

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

        return x

    def clear_cache(self):
        # self.text_cond, self.text_uncond = None, None
        self.text_cache = {} # 清空字典

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_lang: bool = False, # cfg for language ids
        drop_text: bool = False,  # cfg for text
        infer_mode: bool = False,
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        language_ids: list[str] | torch.Tensor = None, # 在新逻辑里面，应该在外部就把lang_to_id做好，传进来一个tensor
        return_ctc: bool = False,
        layered: bool = False,
        prompt_ids: torch.Tensor = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        lang_ids_tensor = None

        if self.languages is not None:
            assert language_ids is not None, "language_ids must be provided for multilingual training."
            if isinstance(language_ids, list):
                lang_ids_tensor = torch.tensor(
                    [self.lang_to_id[l] for l in language_ids], 
                    dtype=torch.long, device=text.device
                )
            else:
                lang_ids_tensor = language_ids # 可能是 [b] 或 [b, nt]
                prompt_ids_tensor = prompt_ids

        def get_branch_inputs(d_audio, d_text, d_lang, use_prompt_id=False, prompt_ids=None):
            curr_lang_ids = None
            if lang_ids_tensor is not None and not use_prompt_id:
                curr_lang_ids = lang_ids_tensor.clone()
                # if d_lang and self.drop_lang_in_time:
                #     curr_lang_ids = torch.full_like(curr_lang_ids, self.num_languages)
            elif use_prompt_id and prompt_ids is not None:
                curr_lang_ids = prompt_ids_tensor.clone()

            t_branch = t.clone()
            if self.languages is not None and self.time_infill_lang_type in ["add_only", "time_concat"]:
                # 如果是 [b, nt]，取最后一个 token 代表目标语种
                g_lang_ids = curr_lang_ids if curr_lang_ids.dim() == 1 else curr_lang_ids[:, -1]
                l_emb = self.lang_embed(g_lang_ids)
                if self.time_infill_lang_type == "add_only":
                    # DiT Additive 融合
                    t_branch = t_branch + self.lang_proj(l_emb)
                elif self.time_infill_lang_type == "time_concat":
                    t_branch = self.cond_fusion(torch.cat([t_branch, l_emb], dim=-1))
            
            if d_lang and self.drop_lang_in_time:
                    curr_lang_ids = torch.full_like(curr_lang_ids, self.num_languages)

            x_embed = self.get_input_embed(
                x, cond, text, 
                drop_audio_cond=d_audio, 
                drop_text=d_text, 
                drop_lang=d_lang, 
                cache=cache, 
                audio_mask=mask, 
                lang_ids_tensor=curr_lang_ids,
            )
            return x_embed, t_branch
                
        
        if cfg_infer and not layered and prompt_ids is None:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, t_cond = get_branch_inputs(False, False, False)
            x_uncond, t_uncond = get_branch_inputs(True, True, True)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t_cond, t_uncond), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        elif cfg_infer and not layered and prompt_ids is not None:
            x_cond, t_cond = get_branch_inputs(False, False, False)
            x_uncond, t_uncond = get_branch_inputs(True, True, True, use_prompt_id=True, prompt_ids=prompt_ids)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t_cond, t_uncond), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        elif cfg_infer and layered:
            x_cond, t_cond = get_branch_inputs(False, False, False) # 都不drop
            x_text, t_text = get_branch_inputs(False, False, True) # drop 语种
            x_uncond, t_uncond = get_branch_inputs(True, True, True) # drop 三个
            x = torch.cat((x_cond, x_text, x_uncond), dim=0)
            t = torch.cat((t_cond, t_text, t_uncond), dim=0)
            mask = torch.cat((mask, mask, mask), dim=0) if mask is not None else None
        else:
            x, t = get_branch_inputs(drop_audio_cond, drop_text, drop_lang)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, infer_mode, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope, infer_mode=infer_mode)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        if infer_mode:
            x = x.to(torch.float16)
        output = self.proj_out(x)

        if self.use_ctc and return_ctc:
            ctc_logits = self.ctc_head(x)
            return output, ctc_logits
        return output
