import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
    get_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM, CFM_SFT
from f5_tts.model.utils import get_tokenizer, get_ipa_id
from f5_tts.eval.speaking_rate_predictor import SpeedPredictor

from f5_tts.train.datasets.ipa_tokenizer import PhonemizeTextTokenizer
from f5_tts.train.datasets.ipa_v2_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v2
from f5_tts.train.datasets.ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v3
from f5_tts.train.datasets.ipa_v4_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v4
from f5_tts.train.datasets.ipa_v5_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v5
from f5_tts.train.datasets.ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v6

# import debugpy
# debugpy.listen(('localhost', 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

#accelerator = Accelerator(mixed_precision="fp16")
accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1


rel_path = str(files("f5_tts").joinpath("../../"))


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, type=int)

    parser.add_argument("-sp", "--sp_type", default="utf", type=str)
    parser.add_argument("-ns", "--expnamesp", default=None, type=str)
    parser.add_argument("-cs", "--ckptstepsp", default=10000, type=int)
    parser.add_argument("-r", "--reverse", action="store_true")

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument("-l", "--languages", default="en", help="Comma separated list of languages or 'all'")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--cfg_strength", default=2.0, type=float)
    parser.add_argument("--cfg_schedule", default=None, type=str)
    parser.add_argument("--cfg_decay_time", default=0.6, type=float)
    parser.add_argument("--cfg_strength2", default=0.0, type=float)
    parser.add_argument("--decode_dir", default=None, type=str)
    parser.add_argument("--concat_method", default=1, type=int)
    parser.add_argument("--layered", action="store_true")
    parser.add_argument("--prompt_v", action="store_true")
    parser.add_argument("--drop_text", action="store_true")
    
    parser.add_argument("--cross_lingual", action="store_true")
    parser.add_argument("-rl", "--reference_languages", default=None, help="Comma separated list of languages, length is the same as --languages")
    

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset
    sp_type = args.sp_type
    exp_name_sp = args.expnamesp
    ckpt_step_sp = args.ckptstepsp
    reverse = args.reverse
    cfg_schedule = args.cfg_schedule
    cfg_decay_time = args.cfg_decay_time
    drop_text = args.drop_text
    cross_lingual = args.cross_lingual
    decode_dir = args.decode_dir
    concat_method = args.concat_method
    
    in2lang = { 
        "th":"thai", "id":"indonesian", "vi":"vietnamese", 
        "zh":"chinese", "en":"english",
        "de":"german", "fr":"french", "es":"spanish", "pl":"polish", "it":"italian", "nl":"dutch", "pt":"portuguese",
        "ko":"korean", "ja":"japanese", "ru":"russian",
        "ro":"romanian","hu":"hungarian","cs":"czech","fi":"finnish","hr":"croatian","sk":"slovak","sl":"slovenian","et":"estonian",
        "lt":"lthuanian","bg":"bulgarian","el":"greek","lv":"latvian","mt":"maltese","sv":"swedish","da":"danish",
        "yue":"cantonese", "ca":"catalan"
    }
    lang2in = {value: key for key, value in in2lang.items()}
    if args.languages == "all":
        target_languages = list(in2lang.keys())
    else:
        target_languages = []
        languages_set = args.languages.split(",")
        for language in languages_set:
            if language in in2lang:
                target_languages.append(language)
            elif language in lang2in:
                target_languages.append(lang2in[language])
            else:
                print(f"Not supported {language}")
                continue
    if cross_lingual:
        reference_languages = []
        ref_languages_set = args.reference_languages.split(",")
        for language in ref_languages_set:
            if language in in2lang:
                reference_languages.append(language)
            elif language in lang2in:
                reference_languages.append(lang2in[language])
            else:
                print(f"Not supported {language}")
                continue

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = args.cfg_strength
    cfg_strength2 = args.cfg_strength2
    layered = args.layered
    prompt_v = args.prompt_v
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft
    
    
    sft = OmegaConf.select(model_cfg, "model.sft", default=False)  
    
    
    
    # speedpredictor config
    if sp_type == "pretrained":
        sp_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name_sp}.yaml")))
        mel_spec_kwargs = sp_cfg.model.mel_spec
        sp_arc = sp_cfg.model.arch
        
        model_sp = SpeedPredictor(
            mel_spec_kwargs=mel_spec_kwargs,
            arch_kwargs = sp_arc
        ).to(device)
        
        ckpt_path_sp = rel_path + f"/ckpts/{exp_name_sp}/model_{ckpt_step_sp}.pt"
        if not os.path.exists(ckpt_path_sp):
            ckpt_path_sp = rel_path + f"/{sp_cfg.ckpts.save_dir}/model_{ckpt_step_sp}.pt"
        print(f"Loading Speaking Rate Predictor checkpoints from {ckpt_path_sp}.")
        dtype = torch.float32
        model_sp = load_checkpoint(model_sp, ckpt_path_sp, device, dtype=dtype, use_ema=use_ema)
    else:
        model_sp = None
    
    # Vocoder model
    local = True
    if mel_spec_type == "vocos":
        vocoder_local_path = "my_vocoder/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer, sft=sft)
    print(f"Dataset: {dataset_name}")
    print(f"Vocab size: {vocab_size}")
    if sft:
        # Model
        model = CFM_SFT(
            transformer=model_cls(
                **model_arc, 
                sft=sft,
                text_num_embeds=vocab_size+1,
                mel_dim=n_mel_channels
            ),
            tokenizer=tokenizer,
            mel_spec_kwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(device)
    else:
        model = CFM(
            transformer=model_cls(
                **model_arc, 
                sft=sft,
                text_num_embeds=vocab_size,
                mel_dim=n_mel_channels
            ),
            tokenizer=tokenizer,
            mel_spec_kwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(device)
        # model.transformer.checkpoint_activations = False 
    
        
    ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}"
    if os.path.exists(ckpt_prefix + ".pt"):
        ckpt_path = ckpt_prefix + ".pt"
    elif os.path.exists(ckpt_prefix + ".safetensors"):
        ckpt_path = ckpt_prefix + ".safetensors"
    else:
        ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    dtype = torch.float32 if mel_spec_type == "bigvgan"  else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    # model = accelerator.prepare(model)
    
    lang_to_id = model.transformer.lang_to_id
    text_infill_lang_type = model.transformer.text_infill_lang_type
    time_infill_lang_type = model.transformer.time_infill_lang_type
    tokenizer_class_map = {
        "ipa": PhonemizeTextTokenizer,
        "ipa_v2": PhonemizeTextTokenizer_v2,
        "ipa_v3": PhonemizeTextTokenizer_v3,
        "ipa_v5": PhonemizeTextTokenizer_v5,
        "ipa_v6": PhonemizeTextTokenizer_v6,
    }
    for i, in_language in enumerate(target_languages):
        ipa_tokenizer = None
        ref_ipa_tokenizer = None
        ref_language = None
        in_language_idx = lang_to_id.get(in_language, len(lang_to_id))
        if in_language_idx == len(lang_to_id):
            print(f"Not supported language: {in_language}, id will set to <unk>.")
        if tokenizer in tokenizer_class_map:
            ipa_id = get_ipa_id(in_language) 
            tokenizer_class = tokenizer_class_map[tokenizer]
            ipa_tokenizer = tokenizer_class(language=ipa_id, with_stress=True)
            # print("词表不加重音，否则请取消注释，包括下面")
            if cross_lingual:
                ref_language= reference_languages[i]
                ref_ipa_id = get_ipa_id(ref_language)
                ref_ipa_tokenizer = tokenizer_class(language=ref_ipa_id, with_stress=True)
                ref_language_idx = lang_to_id.get(ref_language, len(lang_to_id))
        
        if testset == "ls_pc_test_clean":
            data_dir = "/data"
            metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
            librispeech_test_clean_path = "/inspire/dataset/libritts/v1/test-clean"  # test-clean path
            metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

        elif testset == "seedtts_test_zh":
            data_dir = rel_path + "/data/seedtts_testset/zh"
            metalst = data_dir + "/meta.lst"
            metainfo = get_seedtts_testset_metainfo(metalst)

        elif testset == "seedtts_test_en":
            
            data_dir = rel_path + "/data/seedtts_testset/en"
            metalst = data_dir + "/meta.lst"
            metainfo = get_seedtts_testset_metainfo(metalst)
            
        elif testset in ["cv3_eval", "lemas_eval", "lemas_eval_new", "mixed_eval", "mixed_eval_with_gt"]:
            data_dir = rel_path + f"/data/{testset}/zero_shot/{in_language}"
            print(f"Loading {testset} data from: {data_dir}")
            metainfo = get_testset_metainfo(data_dir, in_language, ref_language, drop_text=drop_text)
            
            # task_lang_path = data_dir.split(f"{testset}/")[-1] # 要改！
            # output_dir = f"results/{task_lang_path}/wavs" # 生成到 results/zero_shot/bg/wavs
            # print(f"Override output_dir to: {output_dir}")


        # path to save genereted wavs
        if decode_dir is not None:
            output_dir = decode_dir
        else:
            if not layered:
                output_dir = (
                    f"{rel_path}/"
                    f"results/{exp_name}_{ckpt_step}/{testset}/"
                    f"seed{seed}_concat{concat_method}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
                    f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
                    f"_cfg{cfg_strength}_speed{speed}"
                    f"{'_gt-dur' if use_truth_duration else ''}"
                    f"{'_no-ref-audio' if no_ref_audio else ''}"
                    "zero_shot"
                )
            else:
                output_dir = (
                    f"{rel_path}/"
                    f"results/{exp_name}_{ckpt_step}/{testset}/"
                    f"seed{seed}_concat{concat_method}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
                    f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
                    f"_cfgI{cfg_strength}_cfgII{cfg_strength2}_speed{speed}"
                    f"{'_gt-dur' if use_truth_duration else ''}"
                    f"{'_no-ref-audio' if no_ref_audio else ''}"
                    "zero_shot"
                )
        if testset in ["cv3_eval","lemas_eval", "lemas_eval_new", "mixed_eval", "mixed_eval_with_gt"]:
            if cross_lingual:
                output_dir += f"/{ref_language}_{in_language}/wavs"
            else:
                output_dir += f"/{in_language}/wavs"
                
        print(f"will be saved to:{output_dir}")
        
        
        if not os.path.exists(output_dir) and accelerator.is_main_process:
            os.makedirs(output_dir)
    # -------------------------------------------------#

        prompts_all = get_inference_prompt(
            metainfo,
            speed=speed,
            tokenizer=tokenizer,
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
            mel_spec_type=mel_spec_type,
            target_rms=target_rms,
            use_truth_duration=use_truth_duration,
            infer_batch_size=infer_batch_size,
            language=in_language,
            ipa_tokenizer=ipa_tokenizer,
            normalize_text=args.normalize_text,
            drop_text=drop_text,
            reverse=reverse,
            sp_type=sp_type,
            model_sp=model_sp,
            device=device,
            ref_language=ref_language,
            ref_ipa_tokenizer=ref_ipa_tokenizer,
        )
 
        # start batch inference
        accelerator.wait_for_everyone()
        start = time.time()

        with accelerator.split_between_processes(prompts_all) as prompts:
            for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
                utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list, ref_text_lens, gen_text_lens = prompt
                ref_mels = ref_mels.to(device)
                ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
                total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)
                
                batch_lang_ids = []
                batch_prompt_lang_ids = []
                for r_len, g_len in zip(ref_text_lens, gen_text_lens):
                    if cross_lingual and not drop_text and text_infill_lang_type in ["token_concat", "ada"]:
                        if concat_method == 1:
                            ids = [ref_language_idx] * r_len + [in_language_idx] * g_len
                        elif concat_method == 2:
                            ids =  [in_language_idx] * (r_len + g_len)
                        elif concat_method == 3:
                            unk_idx = len(lang_to_id)
                            ids = [unk_idx] * r_len + [in_language_idx] * g_len
                    else:
                        ids = [in_language_idx] * r_len + [in_language_idx] * g_len
                    if drop_text:
                        ids = [in_language_idx]
                    batch_lang_ids.append(torch.tensor(ids))
                    if prompt_v:
                        prompt_ids = [ref_language_idx] * (r_len + g_len)
                        batch_prompt_lang_ids.append(torch.tensor(prompt_ids))
                lang_ids_tensor = pad_sequence(batch_lang_ids, batch_first=True, padding_value=in_language_idx).to(device)
                if prompt_v:
                    prompt_lang_ids_tensor = pad_sequence(batch_prompt_lang_ids, batch_first=True, padding_value=ref_language_idx).to(device)
                else:
                    prompt_lang_ids_tensor = None

                with torch.inference_mode():
                    generated, _ = model.sample(
                        cond=ref_mels,
                        text=final_text_list,
                        duration=total_mel_lens,
                        lens=ref_mel_lens,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        no_ref_audio=no_ref_audio,
                        seed=seed,
                        language_ids=lang_ids_tensor,
                        cfg_schedule=cfg_schedule,
                        cfg_decay_time=cfg_decay_time,
                        reverse=reverse,
                        cfg_strength2=cfg_strength2,
                        layered=layered,
                        prompt_ids=prompt_lang_ids_tensor,
                    )
                    # Final result
                    # def strong_asymptotic_saturation(mel, threshold=2.8, limit=3.5, start_bin=60):
                    #     mel_high = mel[:, :, start_bin:]
                    #     def apply_limit(x, t, m):
                    #         # 只有超过threshold的部分才进入tanh
                    #         margin = m - t
                    #         return torch.where(
                    #             x < t,
                    #             x,
                    #             t + margin * torch.tanh((x - t) / margin)
                    #         )
                    #     mel[:, :, start_bin:] = apply_limit(mel_high, threshold, limit)
                    #     return mel
                    
                    # generated = strong_asymptotic_saturation(generated, threshold=2.5, limit=3.5)
                    
                    for i, gen in enumerate(generated):
                        
                        if reverse:
                            gen = gen[: total_mel_lens[i] - ref_mel_lens[i], :].unsqueeze(0)
                        else:
                            gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                            # gen = gen[ : total_mel_lens[i], :].unsqueeze(0) # 测试用
                        gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                        if mel_spec_type == "vocos":
                            generated_wave = vocoder.decode(gen_mel_spec).cpu()
                        elif mel_spec_type == "bigvgan":
                            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                        if ref_rms_list[i] < target_rms:
                            generated_wave = generated_wave * ref_rms_list[i] / target_rms
                        torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            timediff = time.time() - start
            print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
