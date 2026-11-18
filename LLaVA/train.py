"""Trains the projector between the d2g encoder and a frozen language model on the captions of the
plans. The run is logged to Weights & Biases, with its settings, the training loss of every update and
the validation loss every so often, beside the loss of a blind model shown another plan's patches.

The run is kept beside the encoder run whose embeddings it reads, at
``<encoder run>/llava/<language model>/<run name>``, with a folder per projector it trains named
``<layer>-<dataset>``. ``<update>.pt`` there holds the projector's weights at every validated update,
with the language model they speak to.

    uv run train.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --language-model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from collator import IMAGE_PATCH, ConversationCollator
from dataset import PlanConversations
from model import LLaVA

CAPTIONING = "jotamunz/synthetic-floorplans-captioning"
WANDB_PROJECT = "record-llava"


def main():
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments.add_argument("--embeddings", type=Path, required=True, help="the folder containing the pregenerated embeddings")
    arguments.add_argument("--language-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    arguments.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="a unique name for the run")
    arguments.add_argument("--seed", type=int, default=0)
    arguments.add_argument("--batch-size", type=int, default=256, help="plans per update")
    arguments.add_argument("--micro-batch-size", type=int, default=32, help="plans per forward pass, accumulated into the batch")
    arguments.add_argument("--learning-rate", type=float, default=1e-3)
    arguments.add_argument("--warmup", type=float, default=0.03, help="share of updates the learning rate rises over")
    arguments.add_argument("--epochs", type=int, default=1)
    arguments.add_argument("--validate-every", type=int, default=50, help="updates per checkpoint")
    arguments.add_argument("--workers", type=int, default=4, help="processes loading plans")
    arguments.add_argument("--overfit", action="store_true", help="train and validate on the first batch of training plans alone")
    settings = arguments.parse_args()
    if settings.batch_size % settings.micro_batch_size != 0:
        raise ValueError("a batch of %d plans does not cut into micro-batches of %d" % (settings.batch_size, settings.micro_batch_size))
    micro_batches_per_update = settings.batch_size // settings.micro_batch_size

    torch.manual_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_run = settings.embeddings.resolve().parent
    run_dir = encoder_run / "llava" / settings.language_model.split("/")[-1] / settings.run_name
    patches_dir = run_dir / ("patches-" + CAPTIONING.split("/")[-1])
    patches_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.language_model)
    language_model = AutoModelForCausalLM.from_pretrained(settings.language_model, dtype=torch.bfloat16)
    train_dataset = PlanConversations("train", settings.embeddings, CAPTIONING)
    val_dataset = PlanConversations("val", settings.embeddings, CAPTIONING)
    if settings.overfit:
        train_dataset = Subset(train_dataset, range(settings.batch_size))
        val_dataset = train_dataset
    model = LLaVA(language_model, tokenizer.convert_tokens_to_ids(IMAGE_PATCH), patch_embedding=train_dataset[0].patches.shape[-1]).to(device)

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=settings.micro_batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            collate_fn=ConversationCollator(tokenizer),
            num_workers=settings.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=settings.workers > 0,
        )

    train_loader, val_loader = loader(train_dataset, shuffle=True), loader(val_dataset, shuffle=False)
    updates_per_epoch = len(train_loader) // micro_batches_per_update
    total_updates = settings.epochs * updates_per_epoch

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=settings.learning_rate, weight_decay=0.0)
    schedule = get_cosine_schedule_with_warmup(optimizer, math.ceil(settings.warmup * total_updates), total_updates)

    def on_device(batch):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    @torch.no_grad()
    def validate(run, update):
        """Logs the validation loss of the model against a blind model that sees the patches of the previous plan."""
        val_losses, blind_losses = [], []
        for micro_batch in val_loader:
            micro_batch = on_device(micro_batch)
            val_losses.append(model(**micro_batch).loss.item())
            previous_plan_patches = micro_batch["patches"].roll(1, dims=0)
            blind_losses.append(model(**(micro_batch | {"patches": previous_plan_patches})).loss.item())
        val_loss, blind_loss = sum(val_losses) / len(val_losses), sum(blind_losses) / len(blind_losses)
        run.log({"val/loss": val_loss, "val/blind_loss": blind_loss}, step=update)
        torch.save({"language_model": settings.language_model, "projector": model.projector.state_dict()}, patches_dir / ("%d.pt" % update))
        print("epoch %d | update %d/%d | val loss %.4f | blind %.4f" % (math.ceil(update / updates_per_epoch), update, total_updates, val_loss, blind_loss))

    print("training a projector for %s: %d updates of %d plans" % (settings.language_model, total_updates, settings.batch_size))
    with wandb.init(project=WANDB_PROJECT, name=settings.run_name, config=vars(settings), dir=run_dir) as run:
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

                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                update += 1
                run.log({"train/loss": float(update_loss), "train/learning_rate": schedule.get_last_lr()[0]}, step=update)
                update_loss = 0.0

                if update % settings.validate_every == 0 or update == total_updates:
                    validate(run, update)


if __name__ == "__main__":
    main()
