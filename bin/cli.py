from ser.data import get_train_dataloader, get_validation_dataloader
from ser.model import get_model
from ser.train import train_model
from ser.constants import RESULTS_DIR

from pathlib import Path
import torch
import json
import os


import typer

main = typer.Typer()

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "--e", "--epochs", help="Number of epochs to train for."
    ),
    batch_size: int = typer.Option(
        1000, "--bs", "--batch-size", help="Number of training examples used in one iteration."
    ),
    learning_rate: float = typer.Option(
        0.01, "--lr", "--learning-rate", help="Step size at each iteration"
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    params = {
        "name": name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    # create folder for parameters if it doesn't exist
    os.makedirs(f"{RESULTS_DIR}/{name}", exist_ok=True)
    #save unique txt file for each experiment
    with open(f"{RESULTS_DIR}/{name}/parameters.txt", 'w') as f:
        json.dump(params, f)

    # load model
    model = get_model(device)

    # dataloaders
    training_dataloader = get_train_dataloader(batch_size)
    val_dataloader = get_validation_dataloader(batch_size)

    # train model, and save model to pt file, and highest accuracy
    train_model(params, device, training_dataloader, val_dataloader, model)


@main.command()
def infer():
    print("This is where the inference code will go")
