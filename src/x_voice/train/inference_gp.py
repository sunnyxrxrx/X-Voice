import os
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from x_voice.model import CFM, Trainer
from x_voice.model.dataset import load_dataset_gp
from x_voice.model.utils import get_tokenizer
from x_voice.model.inferencer_gp import Inferencer_gp

# import debugpy
# debugpy.listen(('localhost', 771))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

PROJECT_ROOT = Path(str(files("x_voice").joinpath("../.."))).resolve()

os.chdir(str(PROJECT_ROOT))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("x_voice").joinpath("configs")), config_name="F5TTS_v1_Base_multilingual_full_catada_stress_qyl_test_infer")
def main(inference_cfg):
    cfg = OmegaConf.load(str(files("x_voice").joinpath(f"configs/{inference_cfg.source.expname}.yaml")))
    model_cls = hydra.utils.get_class(f"x_voice.model.{cfg.model.backbone}")
    model_arc = cfg.model.arch
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type

    gen_root = inference_cfg.infer.gen_root
    target_dir = Path(f"{inference_cfg.datasets.name}_gen")
    if gen_root is None:
        root_path = PROJECT_ROOT / target_dir
    else:
        gen_root_path = Path(gen_root)
        if not gen_root_path.is_absolute():
            raise ValueError("gen_root must be an absolute path when it is not null.")
        root_path = gen_root_path / target_dir

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    cfg.model.arch.attn_backend = inference_cfg.attn.attn_backend
    cfg.model.arch.attn_mask_enabled = inference_cfg.attn.attn_mask_enabled
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    
    ckpt_prefix = f"ckpts/{inference_cfg.source.expname}/model_{inference_cfg.source.ckptstep}"
    if os.path.exists(ckpt_prefix + ".safetensors"):
        source_ckpt = ckpt_prefix + ".safetensors"
    elif os.path.exists(ckpt_prefix + ".pt"):
        source_ckpt = ckpt_prefix + ".pt"
    else:
        print("Loading from self-organized training checkpoints rather than released pretrained.")
        ckpt_prefix = f"{cfg.ckpts.save_dir}/model_{inference_cfg.source.ckptstep}"
        if os.path.exists(ckpt_prefix + ".safetensors"):
            source_ckpt = ckpt_prefix + ".safetensors"
        elif os.path.exists(ckpt_prefix + ".pt"):
            source_ckpt = ckpt_prefix + ".pt"
        else:
            raise ValueError("The checkpoint does not exist or cannot be found in given location.")
    
    inferencer = Inferencer_gp(
        model=model,
        checkpoint_path=source_ckpt,
        root_path=str(root_path),
        batch_size_per_gpu=inference_cfg.datasets.batch_size_per_gpu,
        batch_size_type=inference_cfg.datasets.batch_size_type,
        max_samples=inference_cfg.datasets.max_samples,
        mel_spec_type=mel_spec_type,
        tokenizer=tokenizer,
        nfe_step=inference_cfg.infer.nfe_step,
        cfg_strength=inference_cfg.infer.cfg_strength,
        sway_sampling_coef=inference_cfg.infer.sway_sampling_coef,
        save_wav=inference_cfg.infer.save_wav,
    )

    inference_dataset = load_dataset_gp(inference_cfg.datasets.name, root_dir=inference_cfg.datasets.root_dir, tokenizer=tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    inferencer.inference(
        inference_dataset,
        num_workers=inference_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
    
# accelerate launch src/x_voice/train/inference_gp.py --config-name F5TTS_v1_Base_multilingual_full_catada_stress_qyl_test_infer.yaml
