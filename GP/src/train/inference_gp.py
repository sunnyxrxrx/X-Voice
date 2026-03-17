# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset_gp
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base_debug_Emilia_gp_inference")
def main(inference_cfg):
    cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{inference_cfg.source.expname}.yaml")))
    model_cls = globals()[cfg.model.backbone]
    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    
    exp_name = f"{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    
    ckpt_prefix = abs_path + f"/ckpts/{inference_cfg.source.expname}/model_{inference_cfg.source.ckptstep}"
    if os.path.exists(ckpt_prefix + ".safetensors"):
        source_ckpt = ckpt_prefix + ".safetensors"
    elif os.path.exists(ckpt_prefix + ".pt"):
        source_ckpt = ckpt_prefix + ".pt"
    else:
        print("Loading from self-organized training checkpoints rather than released pretrained.")
        ckpt_prefix = abs_path + f"/{cfg.ckpts.save_dir}/model_{inference_cfg.source.ckptstep}"
        if os.path.exists(ckpt_prefix + ".safetensors"):
            source_ckpt = ckpt_prefix + ".safetensors"
        elif os.path.exists(ckpt_prefix + ".pt"):
            source_ckpt = ckpt_prefix + ".pt"
        else:
            raise ValueError("The checkpoint does not exist or cannot be found in given location.")
    
    inferencer = Inferencer_gp(
        model=model,
        checkpoint_path=source_ckpt,
        root_path=f"{abs_path}/{inference_cfg.datasets.name}_gen",
        batch_size_per_gpu=inference_cfg.datasets.batch_size_per_gpu,
        batch_size_type=inference_cfg.datasets.batch_size_type,
        max_samples=inference_cfg.datasets.max_samples,
        mel_spec_type=mel_spec_type,
        frac_lengths_mask=inference_cfg.source.frac_lengths_mask,
        tokenizer=tokenizer,
        nfe_step=inference_cfg.infer.nfe_step,
        cfg_strength=inference_cfg.infer.cfg_strength,
        sway_sampling_coef=inference_cfg.infer.sway_sampling_coef
    )

    inference_dataset = load_dataset_gp(inference_cfg.datasets.name, inference_cfg.datasets.tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    inferencer.inference(
        inference_dataset,
        num_workers=inference_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
