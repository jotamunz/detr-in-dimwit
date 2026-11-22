"""The training every stage shares. A stage builds its model, says what of it learns through the optimizer
it hands over and how to save it; the rest is the same.

A run is logged to Weights & Biases, with its settings, the training loss of every update and the
validation loss every so often, beside the loss of a blind model shown another plan's patches. It is
kept beside the encoder run whose embeddings it reads, at
``<encoder run>/llava/<language model>/<run name>``, with a folder for what it trains named
``<layer>-<dataset>``.
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup

WANDB_PROJECT = "record-llava"


def arguments(description):
    """The settings every stage takes, to which a stage adds its own."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True, help="the folder containing the pregenerated embeddings")
    parser.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="a unique name for the run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256, help="plans per update")
    parser.add_argument("--micro-batch-size", type=int, default=32, help="plans per forward pass, accumulated into the batch")
    parser.add_argument("--warmup", type=float, default=0.03, help="share of updates the learning rate rises over")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=50, help="updates per checkpoint")
    parser.add_argument("--workers", type=int, default=4, help="processes loading plans")
    parser.add_argument("--overfit", action="store_true", help="train and validate on the first batch of training plans alone")
    return parser


def parse(parser):
    settings = parser.parse_args()
    if settings.batch_size % settings.micro_batch_size != 0:
        raise ValueError("a batch of %d plans does not cut into micro-batches of %d" % (settings.batch_size, settings.micro_batch_size))
    torch.manual_seed(settings.seed)
    return settings


def run_dir(settings):
    return settings.embeddings.resolve().parent / "llava" / settings.language_model.split("/")[-1] / settings.run_name


def train(settings, model, optimizer, collator, train_dataset, val_dataset, save):
    """Train the parameters ``optimizer`` holds, on a learning rate that rises over the warmup and then
    falls along a cosine, and save checkpoints with ``save(update)``.
    """
    micro_batches_per_update = settings.batch_size // settings.micro_batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainable_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    if settings.overfit:
        train_dataset = Subset(train_dataset, range(settings.batch_size))
        val_dataset = train_dataset

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=settings.micro_batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=collator,
            num_workers=settings.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=settings.workers > 0,
        )

    train_loader, val_loader = loader(train_dataset, shuffle=True), loader(val_dataset, shuffle=False)
    updates_per_epoch = len(train_loader) // micro_batches_per_update
    total_updates = settings.epochs * updates_per_epoch
    schedule = get_cosine_schedule_with_warmup(optimizer, math.ceil(settings.warmup * total_updates), total_updates)

    def on_device(batch):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    @torch.no_grad()
    def validate(run, update):
        """Logs the validation loss of the model against a blind model that sees the patches of the previous plan."""
        model.eval()
        val_losses, blind_losses = [], []
        for micro_batch in val_loader:
            micro_batch = on_device(micro_batch)
            val_losses.append(model(**micro_batch).loss.item())
            previous_plan_patches = micro_batch["patches"].roll(1, dims=0)
            blind_losses.append(model(**(micro_batch | {"patches": previous_plan_patches})).loss.item())
        val_loss, blind_loss = sum(val_losses) / len(val_losses), sum(blind_losses) / len(blind_losses)
        run.log({"val/loss": val_loss, "val/blind_loss": blind_loss}, step=update)
        print("epoch %d | update %d/%d | val loss %.4f | blind %.4f" % (math.ceil(update / updates_per_epoch), update, total_updates, val_loss, blind_loss))
        model.train()

    print("training %d updates of %d plans each" % (total_updates, settings.batch_size))
    directory = run_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    with wandb.init(project=WANDB_PROJECT, name=settings.run_name, config=vars(settings), dir=directory) as run:
        update = 0
        validate(run, update)
        for _ in range(settings.epochs):
            update_loss = 0.0
            for micro_batch_number, micro_batch in enumerate(train_loader, start=1):
                # Discard the last micro-batches of an epoch if they do not fill a batch.
                if micro_batch_number > updates_per_epoch * micro_batches_per_update:
                    break
                loss = model(**on_device(micro_batch)).loss / micro_batches_per_update
                loss.backward()
                update_loss += loss.detach()
                if micro_batch_number % micro_batches_per_update != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                update += 1
                run.log({"train/loss": float(update_loss), "train/learning_rate": schedule.get_last_lr()[0]}, step=update)
                update_loss = 0.0

                if update % settings.validate_every == 0 or update == total_updates:
                    validate(run, update)
                    save(update)
