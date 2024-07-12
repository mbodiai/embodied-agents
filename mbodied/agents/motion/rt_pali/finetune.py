import json
import os

from datasets import load_dataset
from huggingface_hub import login
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from mbodied.agents.motion.rt_pali.data.dataloader import (
    CustomDataset,
    MyDataModule,
    compute_statistics,
    scale_pose_data,
)
from mbodied.agents.motion.rt_pali.model.model import VLAModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main() -> None:
    """Main function to train the VLAModel using the PyTorch Lightning framework.

    This function performs the following steps:
    1. Load environmental variables and API keys from `env.json`.
    2. Authenticate with Hugging Face and Weights & Biases.
    3. Load dataset and compute statistics.
    4. Initialize dataloaders and datasets.
    5. Set up logging and checkpoints.
    6. Train the model using PyTorch Lightning Trainer.
    
    Raises:
        ValueError: If the Hugging Face access token or Weights & Biases API key is not found in `env.json`.
    """
    with open('env.json') as f:
        env_data = json.load(f)
        token = env_data.get('hugging_face_access_token')
        wandb_api_key = env_data.get('wandb_api_key')

    if not token:
        raise ValueError("Access token not found in env.json")
    if not wandb_api_key:
        raise ValueError("Weights & Biases API key not found in env.json")

    login(token=token)
    wandb.login(key=wandb_api_key)

    # Parameters
    batch_size = 16

    # Initialize the model
    model = VLAModel()

    # Prepare your datasets and dataloaders
    ds = load_dataset("mbodiai/oxe_utokyo_xarm_pick_place")['default']

    # subset_size = 100  # Define the size of the subset you want to take
    # ds = ds.select(range(subset_size))
    statistics = compute_statistics(ds)

    # Create the CustomDataset with precomputed statistics
    dataset = CustomDataset(ds, statistics, transform=scale_pose_data)

    data_module = MyDataModule(dataset=dataset, batch_size=batch_size)

    # Set up logging and checkpoints
    wandb_logger = WandbLogger(
        project="rt_pali", dir='mbodied/agents/motion/rt_pali/wandb_logs')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,  # Save all checkpoints
        save_last=False,  # Don't save the last checkpoint
        every_n_epochs=1,  # Save every epoch
        dirpath="mbodied/agents/motion/rt_pali/checkpoints",
        filename="ra_pali_gemma-{epoch:02d}",
    )

    # Trainer
    trainer = Trainer(
        max_steps=20000,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accelerator='gpu',
        devices=[3],
        log_every_n_steps=10,
        accumulate_grad_batches=64,  # Update every 64*16=1024 samples
    )

    # Fit the model using the data module
    trainer.fit(model,
                datamodule=data_module,
                # Uncomment the line below to resume training from a checkpoint
                # ckpt_path="/home/user/tilak/rt_vit_gpt2/checkpoints/ra_pali_gemma-epoch=09-train_loss=1.48.ckpt"
                )


if __name__ == "__main__":
    main()
