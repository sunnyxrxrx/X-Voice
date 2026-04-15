# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from srp.model import SpeedPredictor, Trainer
from srp.model.dataset import load_dataset

os.chdir(str(files("srp").joinpath("../..")))

@hydra.main(version_base="1.3", config_path=str(files("srp").joinpath("configs")), config_name=None)
def main(cfg):
    exp_name = f"{cfg.model.name}_{cfg.datasets.name}_{cfg.model.loss}"
    wandb_resume_id = None

    # set model
    model = SpeedPredictor(
        mel_spec_kwargs=cfg.model.mel_spec,
        loss_type=cfg.model.loss,
        arch_kwargs=cfg.model.arch,
        sigma_factor=cfg.model.get('gce_sigma', 1),
        silence_prob=cfg.model.get("silence_prob", 0.0),
        silence_ratio_min=cfg.model.get("silence_ratio_min", 0.2),
        silence_ratio_max=cfg.model.get("silence_ratio_max", 0.8),
    )
    # init trainer
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("srp").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="SpeedPredictor",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=cfg.ckpts.log_samples,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
    )

    train_dataset = load_dataset(
        cfg.datasets.name, 
        mel_spec_kwargs=cfg.model.mel_spec,
        split="train",
    )
    val_dataset = load_dataset(
        cfg.datasets.name, 
        mel_spec_kwargs=cfg.model.mel_spec,
        split="val",
    )

    trainer.train(
        train_dataset,
        val_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

if __name__ == "__main__":
    main()

# accelerate launch src/train/train.py --config-name SpeedPredict_Multilingual.yaml
