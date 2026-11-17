"""The floor plans and the conversations about them, each plan as the patches the d2g encoder made of
its drawing.

The encoder has already read every plan: ``d2g/out/embeddings/{part}/patches.npy`` holds
``(plans, patches, embedding)`` in float16 and ``seeds.npy`` the seed of every row. The conversations
are those of https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning, whose ``id`` is
the same seed. A split is one part or, for train, the parts ``train-00`` to ``train-03``, and both
sides keep their plans in seed order, so row ``i`` of the one is row ``i`` of the other.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

CAPTIONING = "jotamunz/synthetic-floorplans-captioning"
EMBEDDINGS = Path(__file__).resolve().parent.parent / "d2g" / "out" / "embeddings"


@dataclass
class PlanConversation:
    seed: int
    """The seed used to create the plan"""
    patches: np.ndarray
    """``(patches, embedding)``, the encoder embedding of the plan"""
    conversation: list
    """The conversation turns as: ``{"from": "human" | "gpt", "value": ...}``, with ``<image>``
    as a placeholder for the patches"""


class PlanConversations(Dataset):
    """The conversations for the plans of a split."""

    def __init__(self, split, captioning=CAPTIONING, embeddings=EMBEDDINGS):
        """
        :param split: ``train``, ``val`` or ``test``
        :param captioning: the Hugging Face dataset for captioning
        :param embeddings: the folder containing the embeddings
        """
        # Load all the parts of the split
        self.parts = sorted(part for part in embeddings.iterdir() if part.name.startswith(split))
        if not self.parts:
            raise FileNotFoundError("no embeddings of %s in %s" % (split, embeddings))
        seeds_per_part = [np.load(part / "seeds.npy") for part in self.parts]
        self.part_starts = np.cumsum([0] + [len(part_seeds) for part_seeds in seeds_per_part])
        self.seeds = np.concatenate(seeds_per_part)

        # Download the captioning dataset and read the conversations of the split, which are in the same order as the plans.
        snapshot = Path(snapshot_download(captioning, repo_type="dataset", allow_patterns="data/%s-*.parquet" % split))
        shards = sorted(snapshot.glob("data/%s-*.parquet" % split))
        captions = pa.concat_tables(pq.read_table(shard, columns=["id", "conversations"]) for shard in shards)
        
        # Check that the conversation ids match the plan seeds.
        if not np.array_equal(np.array(captions["id"].to_pylist(), dtype=np.int64), self.seeds):
            raise ValueError("the conversations in %s don't match the plans in %s" % (split, embeddings))
        self.conversations = captions["conversations"].to_pylist()

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, plan):
        # Mapped per plan: a map held by the dataset would be pickled whole into every spawned loader worker.
        part = np.searchsorted(self.part_starts, plan, side="right") - 1
        patches = np.load(self.parts[part] / "patches.npy", mmap_mode="r")[plan - self.part_starts[part]]
        return PlanConversation(int(self.seeds[plan]), np.array(patches), self.conversations[plan])
