"""Stage 1: trains the projector between the d2g encoder and a frozen language model on the captions of
the plans.

The projector is kept in the run's ``patches-<dataset>`` folder, where ``<update>.pt`` holds its weights
at every validated update, with the language model they speak to.

    uv run train_projector.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --language-model Qwen/Qwen2.5-0.5B-Instruct
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import train
from collator import IMAGE_PATCH, ConversationCollator
from dataset import PlanConversations
from model import LLaVA

DATASET_CAPTIONING = "jotamunz/synthetic-floorplans-captioning"


def main():
    parser = train.arguments(__doc__)
    parser.add_argument("--language-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    settings = train.parse(parser)

    tokenizer = AutoTokenizer.from_pretrained(settings.language_model)
    language_model = AutoModelForCausalLM.from_pretrained(settings.language_model, dtype=torch.bfloat16).requires_grad_(False)
    train_dataset = PlanConversations("train", settings.embeddings, DATASET_CAPTIONING)
    val_dataset = PlanConversations("val", settings.embeddings, DATASET_CAPTIONING)
    model = LLaVA(language_model, tokenizer.convert_tokens_to_ids(IMAGE_PATCH), patch_embedding=train_dataset[0].patches.shape[-1])
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=settings.learning_rate, weight_decay=0.0)

    projector_dir = train.run_dir(settings) / ("patches-" + DATASET_CAPTIONING.split("/")[-1])
    projector_dir.mkdir(parents=True, exist_ok=True)

    def save(update):
        torch.save({"language_model": settings.language_model, "projector": model.projector.state_dict()}, projector_dir / ("%d.pt" % update))

    print("training a projector for %s" % settings.language_model)
    train.train(settings, model, optimizer, ConversationCollator(tokenizer), train_dataset, val_dataset, save)


if __name__ == "__main__":
    main()
