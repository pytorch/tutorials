"""
Example training code for ``ax_multiobjective_nas_tutorial.py``
"""

import argparse
import logging
import os
import sys
import time
import warnings

import torch
from IPython.utils import io
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

warnings.filterwarnings("ignore")  # Disable data logger warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  # Disable GPU/TPU prints


def parse_args():
    parser = argparse.ArgumentParser(description="train mnist")
    parser.add_argument(
        "--log_path", type=str, required=True, help="dir to place tensorboard logs from all trials"
    )
    parser.add_argument(
        "--hidden_size_1", type=int, required=True, help="hidden size layer 1"
    )
    parser.add_argument(
        "--hidden_size_2", type=int, required=True, help="hidden size layer 2"
    )
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs")
    parser.add_argument("--dropout", type=float, required=True, help="dropout probability")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    return parser.parse_args()

args = parse_args()

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class MnistModel(LightningModule):
    def __init__(self):
        super().__init__()

        # Tunable parameters
        self.hidden_size_1 = args.hidden_size_1
        self.hidden_size_2 = args.hidden_size_2
        self.learning_rate = args.learning_rate
        self.dropout = args.dropout
        self.batch_size = args.batch_size

        # Set class attributes
        self.data_dir = PATH_DATASETS

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Create a PyTorch model
        layers = [nn.Flatten()]
        width = channels * width * height
        hidden_layers = [self.hidden_size_1, self.hidden_size_2]
        num_params = 0
        for hidden_size in hidden_layers:
            if hidden_size > 0:
                layers.append(nn.Linear(width, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                num_params += width * hidden_size
                width = hidden_size
        layers.append(nn.Linear(width, self.num_classes))
        num_params += width * self.num_classes

        # Save the model and parameter counts
        self.num_params = num_params
        self.model = nn.Sequential(*layers)  # No need to use Relu for the last layer

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = multiclass_accuracy(preds, y, num_classes=self.num_classes)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)


def run_training_job():

    mnist_model = MnistModel()

    # Initialize a trainer (don't log anything since things get so slow...)
    trainer = Trainer(
        logger=False,
        max_epochs=args.epochs,
        enable_progress_bar=False,
        deterministic=True,  # Do we want a bit of noise?
        default_root_dir=args.log_path,
    )

    logger = pl_loggers.TensorBoardLogger(args.log_path)

    print(f"Logging to path: {args.log_path}.")

    # Train the model and log time ⚡
    start = time.time()
    trainer.fit(model=mnist_model)
    end = time.time()
    train_time = end - start
    logger.log_metrics({"train_time": end - start})

    # Compute the validation accuracy once and log the score
    with io.capture_output() as captured:
        val_accuracy = trainer.validate()[0]["val_acc"]
    logger.log_metrics({"val_acc": val_accuracy})

    # Log the number of model parameters
    num_params = trainer.model.num_params
    logger.log_metrics({"num_params": num_params})

    logger.save()

    # Print outputs
    print(f"train time: {train_time}, val acc: {val_accuracy}, num_params: {num_params}")


if __name__ == "__main__":
    run_training_job()
