import os
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import argparse
import warnings
from datasets.datamodule import CellCycleDataModule, PhaseToFlourDataModule
from models.coatnet import Classifier
from models.morphocycle import MorphoCycle
from lightning.pytorch.callbacks import Callback
import wandb


warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train image to image generator model for cell cycle"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/nvme0n1/Datasets/PCNA/PhaseToFluorNorm16/"
                "reconstructed_QPI_registered/",
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/mnt/nvme0n1/Datasets/PCNA/PhaseToFluorNorm16/"
                "mCh_100Pc_subBkg/",
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1000, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs",
        help="Directory to save the model weights to",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay parameter for optimizer",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/", help="directory to save logs"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="/home/mvries/Documents/GitHub/MorphoCycle/logs/EfficientNet_Morphocycle/3dbugubg/checkpoints/epoch=10-step=352.ckpt",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="ImageToImage_Morphocycle",
        help="Name of the project",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "neptune"],
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to train on",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ImageToImage"],
        default="ImageToImage",
        help="Choice of model.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Stage of training",
    )

    return parser.parse_args()


def train(args):
    print(f"Training {args.model_type} model.")
    # Setting the seed
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=50, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cell_data = PhaseToFlourDataModule(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        batch_size=4,
    )
    cell_data.setup()
    model = MorphoCycle(
        args = args
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project_name,
            log_model=True,
            save_dir=args.log_dir,
        )

    elif args.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.log_dir,
        )

    else:
        raise ValueError(f"Invalid logger {args.logger}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.max_epochs,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_monitor,
            # LogPredictionSamplesCallback()
        ],
        default_root_dir=args.log_dir,
        logger=logger,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, datamodule=cell_data)
    print(f"Finished training model.")
    trainer.test(model=model, datamodule=cell_data)


def test(args):
    print(f"Testing {args.model_type} model.")
    # Setting the seed
    pl.seed_everything(42)
    cell_data = CellCycleDataModule(
        img_dir=args.img_dir,
        batch_size=args.batch_size,
    )
    cell_data.setup(stage="test")
    model = Classifier(
        num_classes=4,
        pretrained=True,
    )
    model.load_from_checkpoint(args.pretrained_model_path)

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project_name,
            log_model=True,
            save_dir=args.log_dir,
        )

    elif args.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.log_dir,
        )

    else:
        raise ValueError(f"Invalid logger {args.logger}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.max_epochs,
        default_root_dir=args.log_dir,
        logger=logger,
        num_sanity_val_steps=0,
    )
    trainer.test(model=model, datamodule=cell_data)


if __name__ == "__main__":
    args = get_args()
    if args.stage == "train":
        train(args)
    elif args.stage == "test":
        test(args)
