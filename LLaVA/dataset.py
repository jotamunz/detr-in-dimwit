"""The floor plans and the conversations about them, each plan as what the d2g model made of its drawing:
the patches its encoder read it into, and the nodes and relationships its decoders wrote down.

The model has already read every plan. Its embeddings lie beside the checkpoints of its run, in
``embeddings-d2g-<corpus>-<size>-<run>-<step>``, where ``{part}/seeds.npy`` holds the seed of every row,
``{part}/patches.npy`` ``(plans, patches, embedding)``, and ``{part}/node_predictions.npy``
``(nodes, embedding)`` the nodes of every plan one after another, plan ``i``'s from row
``node_offsets[i]`` until ``node_offsets[i + 1]``; the relationships the same, in
``relationship_predictions.npy`` and ``relationship_offsets.npy``. All in float16.

The conversations are those of a Hugging Face dataset of the same plans, such as
https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning, whose ``id`` is the same seed.
A split is one part or, for train, the parts ``train-00`` to ``train-03``, and both sides keep their plans
in seed order, so row ``i`` of the one is row ``i`` of the other.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

RECORD_FILES = {
    "nodes": ("node_predictions.npy", "node_offsets.npy"),
    "relationships": ("relationship_predictions.npy", "relationship_offsets.npy"),
}


@dataclass
class PlanConversation:
    seed: int
    """The seed used to create the plan"""
    elements: dict
    """For each layer type: ``patches``, ``nodes`` or ``relationships``, its ``(elements, embedding)``"""
    conversation: list
    """The conversation turns as: ``{"from": "human" | "gpt", "value": ...}``, with ``<image>`` as a
    placeholder for the patches and ``<record>`` for the nodes and relationships"""


class PlanConversations(Dataset):
    """The conversations for the plans of a split."""

    def __init__(self, split, embeddings, dataset, layer_types):
        """
        :param split: ``train``, ``val`` or ``test``
        :param embeddings: the folder containing the embeddings
        :param dataset: the Hugging Face dataset of conversations
        :param layer_types: which of ``patches``, ``nodes`` and ``relationships`` the plans are read as
        """
        self.layer_types = layer_types

        # Load all the parts of the split
        self.parts = sorted(part for part in embeddings.iterdir() if part.name.startswith(split))
        if not self.parts:
            raise FileNotFoundError("no embeddings of %s in %s" % (split, embeddings))
        seeds_per_part = [np.load(part / "seeds.npy") for part in self.parts]
        self.part_starts = np.cumsum([0] + [len(part_seeds) for part_seeds in seeds_per_part])
        self.seeds = np.concatenate(seeds_per_part)
        self.offsets = {
            layer_type: [np.load(part / RECORD_FILES[layer_type][1]) for part in self.parts]
            for layer_type in layer_types if layer_type in RECORD_FILES
        }

        # Download the dataset and read the conversations of the split, which are in the same order as the plans.
        snapshot = Path(snapshot_download(dataset, repo_type="dataset", allow_patterns="data/%s-*.parquet" % split))
        shards = sorted(snapshot.glob("data/%s-*.parquet" % split))
        table = pa.concat_tables(pq.read_table(shard, columns=["id", "conversations"]) for shard in shards)

        # Check that the conversation ids match the plan seeds.
        if not np.array_equal(np.array(table["id"].to_pylist(), dtype=np.int64), self.seeds):
            raise ValueError("the conversations in %s don't match the plans in %s" % (split, embeddings))
        self.conversations = table["conversations"].to_pylist()

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, plan):
        # Mapped per plan: a map held by the dataset would be pickled whole into every spawned loader worker.
        part = np.searchsorted(self.part_starts, plan, side="right") - 1
        row = plan - self.part_starts[part]
        elements = {}
        for layer_type in self.layer_types:
            if layer_type == "patches":
                elements[layer_type] = np.array(np.load(self.parts[part] / "patches.npy", mmap_mode="r")[row])
            else:
                start, end = self.offsets[layer_type][part][row:row + 2]
                elements[layer_type] = np.array(np.load(self.parts[part] / RECORD_FILES[layer_type][0], mmap_mode="r")[start:end])
        return PlanConversation(int(self.seeds[plan]), elements, self.conversations[plan])
