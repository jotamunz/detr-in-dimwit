"""Stage 2: from trained projectors, teaches the language model to answer questions about the plans.
Low-rank adapters (LoRA) on its attention and MLP projections learn beside the projectors, on the
multi-turn conversations of the questions set, while the language model's own weights stay as they are.

The plans are read as the layer types whose projectors it is given, any of the patches, nodes and
relationships; the relationships only beside the nodes, as their projector was trained. Every projector
starts from the last one saved in its folder, and all must speak to the same language model. What it
trains is kept in the run's ``lora-<dataset>`` folder, where ``<update>/`` holds at every validated
update the adapters as peft saves them and ``projector.pt``, the projector of each layer type with the
language model they speak to.

    uv run train_lora.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --patch-projector ../results/runs/<encoder run>/llava/<language model>/<run name>/patches-<dataset> \
        --node-projector ../results/runs/<encoder run>/llava/<language model>/<run name>/nodes-<dataset>
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
    parser.add_argument("--patch-projector", type=Path, help="the folder of a trained patch projector's checkpoints")
    parser.add_argument("--node-projector", type=Path, help="the folder of a trained node projector's checkpoints")
    parser.add_argument("--relationship-projector", type=Path, help="the folder of a trained relationship projector's checkpoints")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--adapters-learning-rate", type=float, default=2e-4)
    parser.add_argument("--projector-learning-rate", type=float, default=2e-5)
    parser.set_defaults(batch_size=128, micro_batch_size=16)
    settings = train.parse(parser)

    projector_folders = {"patches": settings.patch_projector, "nodes": settings.node_projector, "relationships": settings.relationship_projector}
    projector_folders = {layer_type: folder for layer_type, folder in projector_folders.items() if folder is not None}
    if not projector_folders:
        parser.error("give at least one of --patch-projector, --node-projector or --relationship-projector")
    if "relationships" in projector_folders and "nodes" not in projector_folders:
        parser.error("--relationship-projector needs --node-projector")
    layer_types = list(projector_folders)

    projector_checkpoints = {}
    for layer_type, folder in projector_folders.items():
        projector_path = train.latest_checkpoint(folder)
        projector_checkpoints[layer_type] = torch.load(projector_path, map_location="cpu")
        print("starting the %s from %s, a projector for %s" % (layer_type, projector_path, projector_checkpoints[layer_type]["language_model"]))
    language_models = {checkpoint["language_model"] for checkpoint in projector_checkpoints.values()}
    if len(language_models) != 1:
        parser.error("the projectors language models don't match: %s" % ", ".join(sorted(language_models)))
    settings.language_model = language_models.pop()

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
    train_dataset = PlanConversations("train", settings.embeddings, DATASET_QA, layer_types)
    val_dataset = PlanConversations("val", settings.embeddings, DATASET_QA, layer_types)
    placeholders = {layer_type: tokenizer.convert_tokens_to_ids(PLACEHOLDERS[layer_type]) for layer_type in layer_types}
    model = LLaVA(language_model, placeholders, element_embedding=train_dataset[0].elements[layer_types[0]].shape[-1])
    for layer_type, checkpoint in projector_checkpoints.items():
        model.projectors[layer_type].load_state_dict(checkpoint["projector"])
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in language_model.parameters() if parameter.requires_grad], "lr": settings.adapters_learning_rate},
            {"params": model.projectors.parameters(), "lr": settings.projector_learning_rate},
        ],
        weight_decay=0.0,
    )

    lora_dir = train.run_dir(settings) / ("lora-" + DATASET_QA.split("/")[-1])

    def save(update):
        checkpoint_dir = lora_dir / str(update)
        language_model.save_pretrained(checkpoint_dir)
        projectors = {layer_type: projector.state_dict() for layer_type, projector in model.projectors.items()}
        torch.save({"language_model": settings.language_model, "projectors": projectors}, checkpoint_dir / "projector.pt")

    language_model.print_trainable_parameters()
    train.train(settings, model, optimizer, ConversationCollator(tokenizer), train_dataset, val_dataset, save)


if __name__ == "__main__":
    main()
