"""Stage 2: from a trained projector, teaches the language model to answer questions about the plans.
Low-rank adapters (LoRA) on its attention and MLP projections learn beside the projector, on the
multi-turn conversations of the questions set, while the language model's own weights stay as they are.

It starts from the last projector saved in the folder it is given, and keeps what it trains in the
run's ``lora-<dataset>`` folder, where ``<update>/`` holds at every validated update the adapters as
peft saves them and ``projector.pt``, the projector with the language model it speaks to.

    uv run train_lora.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --projector ../results/runs/<encoder run>/llava/<language model>/<run name>/patches-<dataset>
"""

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import train
from collator import PLACEHOLDERS, ConversationCollator
from dataset import PlanConversations
from model import LLaVA

DATASET_QA = "jotamunz/synthetic-floorplans-qa"
ADAPTED = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = train.arguments(__doc__)
    parser.add_argument("--projector", type=Path, required=True, help="the folder of a trained projector's checkpoints")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--adapters-learning-rate", type=float, default=2e-4)
    parser.add_argument("--projector-learning-rate", type=float, default=2e-5)
    parser.set_defaults(batch_size=128, micro_batch_size=16)
    settings = train.parse(parser)

    projector_path = train.latest_checkpoint(settings.projector)
    projector_checkpoint = torch.load(projector_path, map_location="cpu")
    settings.language_model = projector_checkpoint["language_model"]
    print("starting from %s, a projector for %s" % (projector_path, settings.language_model))

    tokenizer = AutoTokenizer.from_pretrained(settings.language_model)
    pretrained = AutoModelForCausalLM.from_pretrained(settings.language_model, dtype=torch.bfloat16)
    adapters = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=ADAPTED,
        task_type="CAUSAL_LM",
    )
    language_model = get_peft_model(pretrained, adapters)
    train_dataset = PlanConversations("train", settings.embeddings, DATASET_QA, ["patches"])
    val_dataset = PlanConversations("val", settings.embeddings, DATASET_QA, ["patches"])
    model = LLaVA(language_model, {"patches": tokenizer.convert_tokens_to_ids(PLACEHOLDERS["patches"])}, element_embedding=train_dataset[0].elements["patches"].shape[-1])
    projector = model.projectors["patches"]
    projector.load_state_dict(projector_checkpoint["projector"])
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in language_model.parameters() if parameter.requires_grad], "lr": settings.adapters_learning_rate},
            {"params": projector.parameters(), "lr": settings.projector_learning_rate},
        ],
        weight_decay=0.0,
    )

    lora_dir = train.run_dir(settings) / ("lora-" + DATASET_QA.split("/")[-1])

    def save(update):
        checkpoint_dir = lora_dir / str(update)
        language_model.save_pretrained(checkpoint_dir)
        torch.save({"language_model": settings.language_model, "projector": projector.state_dict()}, checkpoint_dir / "projector.pt")

    language_model.print_trainable_parameters()
    train.train(settings, model, optimizer, ConversationCollator(tokenizer), train_dataset, val_dataset, save)


if __name__ == "__main__":
    main()
