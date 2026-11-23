"""Aligns one layer type's projector between d2g and a frozen language model: the patches on captions of the
whole plans, the nodes on what single nodes are, the relationships on which nodes they link.

A relationship's answer names its nodes, so the relationships are read beside the nodes, through a node
projector trained before and kept as it is.

The projector is kept in the run's ``<layer_type>-<dataset>`` folder, where ``<update>.pt`` holds its weights
at every validated update, with the language model they speak to.

    uv run train_projector.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --layer-type nodes --language-model Qwen/Qwen2.5-0.5B-Instruct

    uv run train_projector.py --embeddings ../results/runs/<encoder run>/embeddings-d2g-<corpus>-<size>-<run>-<step> \
        --layer-type relationships --node-projector ../results/runs/<encoder run>/llava/<language model>/<run name>/nodes-<dataset>
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import train
from collator import PLACEHOLDERS, ConversationCollator
from dataset import PlanConversations
from model import LLaVA

DATASETS_CAPTIONING = {
    "patches": "jotamunz/synthetic-floorplans-captioning",
    "nodes": "jotamunz/synthetic-floorplans-node-captioning",
    "relationships": "jotamunz/synthetic-floorplans-relationship-captioning",
}


def main():
    parser = train.arguments(__doc__)
    parser.add_argument("--layer-type", choices=DATASETS_CAPTIONING, required=True, help="which projector to train")
    # TODO: Get node projector automatically for relationships
    parser.add_argument("--node-projector", type=Path, help="the folder of a trained node projector's checkpoints")
    parser.add_argument("--language-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    settings = train.parse(parser)
    if (settings.layer_type == "relationships") != (settings.node_projector is not None):
        parser.error("--node-projector is only meant for --layer-type relationships")

    # Relationships are trained with a frozen node projector
    layer_types = ["nodes", "relationships"] if settings.node_projector else [settings.layer_type]
    node_projector_checkpoint = None
    if settings.node_projector:
        node_projector_path = train.latest_checkpoint(settings.node_projector)
        node_projector_checkpoint = torch.load(node_projector_path, map_location="cpu")
        if node_projector_checkpoint["language_model"] != settings.language_model:
            parser.error("the node projector %s was trained for %s, not %s" % (node_projector_path, node_projector_checkpoint["language_model"], settings.language_model))

    tokenizer = AutoTokenizer.from_pretrained(settings.language_model)
    language_model = AutoModelForCausalLM.from_pretrained(settings.language_model, dtype=torch.bfloat16).requires_grad_(False)
    train_dataset = PlanConversations("train", settings.embeddings, DATASETS_CAPTIONING[settings.layer_type], layer_types)
    val_dataset = PlanConversations("val", settings.embeddings, DATASETS_CAPTIONING[settings.layer_type], layer_types)
    placeholders = {layer_type: tokenizer.convert_tokens_to_ids(PLACEHOLDERS[layer_type]) for layer_type in layer_types}
    model = LLaVA(language_model, placeholders, element_embedding=train_dataset[0].elements[settings.layer_type].shape[-1])
    if node_projector_checkpoint is not None:
        model.projectors["nodes"].load_state_dict(node_projector_checkpoint["projector"])
        model.projectors["nodes"].requires_grad_(False)
    projector = model.projectors[settings.layer_type]
    optimizer = torch.optim.AdamW(projector.parameters(), lr=settings.learning_rate, weight_decay=0.0)

    projector_dir = train.run_dir(settings) / ("%s-%s" % (settings.layer_type, DATASETS_CAPTIONING[settings.layer_type].split("/")[-1]))
    projector_dir.mkdir(parents=True, exist_ok=True)

    def save(update):
        torch.save({"language_model": settings.language_model, "projector": projector.state_dict()}, projector_dir / ("%d.pt" % update))

    print("training a %s projector for %s" % (settings.layer_type, settings.language_model))
    train.train(settings, model, optimizer, ConversationCollator(tokenizer), train_dataset, val_dataset, save)


if __name__ == "__main__":
    main()
