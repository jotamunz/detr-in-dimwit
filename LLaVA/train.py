"""Trains the projector between the d2g encoder and a frozen language model on the captions of the
plans. The run is logged to Weights & Biases, with its settings, the training loss of every update and
the validation loss every so often. ``projector.pt`` in OUTPUT_DIR (``LLaVA/out`` unless set) holds
the projector's weights as last validated, with the language model they speak to.

    uv run train.py --language-model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import math
import os
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from collator import IMAGE_PAD, ConversationCollator
from dataset import PlanConversations
from model import LLaVA

OUTPUT = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent / "out"))
WANDB_PROJECT = "record-llava"


def main():
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments.add_argument("--language-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    arguments.add_argument("--batch-size", type=int, default=256, help="plans per update")
    arguments.add_argument("--micro-batch-size", type=int, default=32, help="plans per forward pass, accumulated into the batch")
    arguments.add_argument("--learning-rate", type=float, default=1e-3)
    arguments.add_argument("--warmup", type=float, default=0.03, help="share of updates the learning rate rises over")
    arguments.add_argument("--epochs", type=int, default=1)
    arguments.add_argument("--validate-every", type=int, default=50, help="updates per checkpoint")
    arguments.add_argument("--workers", type=int, default=4, help="processes loading plans")
    settings = arguments.parse_args()
    if settings.batch_size % settings.micro_batch_size != 0:
        raise ValueError("a batch of %d plans does not cut into micro-batches of %d" % (settings.batch_size, settings.micro_batch_size))
    micro_batches_per_update = settings.batch_size // settings.micro_batch_size

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.language_model)
    language_model = AutoModelForCausalLM.from_pretrained(settings.language_model, dtype=torch.bfloat16)
    train_dataset, val_dataset = PlanConversations("train"), PlanConversations("val")
    model = LLaVA(language_model, tokenizer.convert_tokens_to_ids(IMAGE_PAD), patch_embedding=train_dataset[0].patches.shape[-1]).to(device)

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
    def validation_loss():
        losses = [model(**on_device(micro_batch)).loss.item() for micro_batch in val_loader]
        return sum(losses) / len(losses)

    print("training a projector for %s: %d updates of %d plans" % (settings.language_model, total_updates, settings.batch_size))
    with wandb.init(project=WANDB_PROJECT, config=vars(settings), dir=OUTPUT) as run:
        update = 0
        for epoch in range(settings.epochs):
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
                run.log({"train/loss": float(update_loss), "train/learning_rate": schedule.get_last_lr()[0], "epoch": epoch + 1}, step=update)
                update_loss = 0.0

                if update % settings.validate_every == 0 or update == total_updates:
                    val_loss = validation_loss()
                    run.log({"val/loss": val_loss}, step=update)
                    torch.save({"language_model": settings.language_model, "projector": model.projector.state_dict()}, OUTPUT / "projector.pt")
                    print("epoch %d | update %d/%d | val loss %.4f" % (epoch + 1, update, total_updates, val_loss))


if __name__ == "__main__":
    main()
