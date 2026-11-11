"""The synthetic floor plans, as the arrays DimWit lifts them from. Everything else happens in Scala.

The dataset (https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription) ships
its splits as plain repository files: ``{part}_images.npy`` holds ``(N, 512, 512)`` uint8 drawings
(row index = y, white background, dark ink) and ``{part}_labels.jsonl`` one record per line, in the
same order. A split is one part or, for train, the parts ``train-00`` to ``train-08``;
``metadata.json`` lists every part and how many plans it holds.

A record holds its nodes -- each a class and the two ends of a segment, in the pixels of the drawing
-- and its relationships, each naming two nodes by their position.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import constants, hf_hub_download


def parts(repo_id, split):
    """``(part, plans)`` of every part of a split, in seed order."""
    held = json.load(open(_file(repo_id, "metadata.json"), encoding="utf-8"))["splits"]
    return sorted((name, held[name]["plans"]) for name in held if name == split or name.startswith(split + "-"))


def drawings(repo_id, part, canvas):
    """The images of a part at ``canvas`` pixels a side, memory mapped so that only what is read is
    read. A smaller side is the stored one averaged down, every pixel the mean of the block it
    covers, so that a thin line turns grey rather than vanishing.

    The averaging is done once, the first time a part is asked for at a side, and kept beside the
    Hugging Face cache: training reads a part hundreds of times, and the stored drawings are 16
    times the bytes of drawings at a quarter of their side.
    """
    stored_at = Path(_file(repo_id, "%s_images.npy" % part))
    stored = np.load(stored_at, mmap_mode="r")
    side = stored.shape[1]
    if side % canvas != 0:
        raise ValueError("drawings of %d pixels do not average down to %d" % (side, canvas))
    if canvas == side:
        return stored
    # The snapshot folder names the revision, so a new release of the dataset is averaged anew.
    kept = Path(constants.HF_HUB_CACHE).parent / "averaged" / repo_id.replace("/", "--") / stored_at.parent.name / ("%s_images_%d.npy" % (part, canvas))
    if not kept.exists():
        _average(stored, side // canvas, kept)
    return np.load(kept, mmap_mode="r")


def _average(stored, factor, kept):
    """Writes ``stored`` averaged down by ``factor`` to ``kept``, a block of drawings at a time.

    Written under a name of its own and renamed when complete, so that two jobs averaging the same
    part at once each leave a whole file, and neither reads a half-written one.
    """
    kept.parent.mkdir(parents=True, exist_ok=True)
    plans, side = stored.shape[0], stored.shape[1] // factor
    handle, writing = tempfile.mkstemp(dir=kept.parent, suffix=".npy")
    os.close(handle)
    averaged = np.lib.format.open_memmap(writing, mode="w+", dtype=np.uint8, shape=(plans, side, side))
    for start in range(0, plans, 1000):
        held = np.asarray(stored[start:start + 1000], dtype=np.float32)
        blocks = held.reshape(len(held), side, factor, side, factor)
        averaged[start:start + len(held)] = np.rint(blocks.mean(axis=(2, 4))).astype(np.uint8)
    averaged.flush()
    del averaged
    os.replace(writing, kept)


def records(repo_id, part, side, nodes, edges, classes, relationships):
    """``(node_class, start_x, start_y, end_x, end_y, relationship, subject, obj)`` of every plan of a
    part, padded to ``nodes`` nodes and ``edges`` relationships.

    ``classes`` and ``relationships`` name the class and relationship of every id, the first naming
    what pads. A point is held as a fraction of the ``side`` the drawings are stored at.
    """
    class_id = {name: at for at, name in enumerate(classes)}
    relationship_id = {name: at for at, name in enumerate(relationships)}
    with open(_file(repo_id, "%s_labels.jsonl" % part), encoding="utf-8") as labels:
        held = [json.loads(label) for label in labels]

    node_class = np.zeros((len(held), nodes), dtype=np.int32)
    start_x, start_y, end_x, end_y = (np.zeros((len(held), nodes), dtype=np.float32) for _ in range(4))
    relationship = np.zeros((len(held), edges), dtype=np.int32)
    subject, obj = (np.zeros((len(held), edges), dtype=np.int32) for _ in range(2))

    for plan, record in enumerate(held):
        if len(record["nodes"]) > nodes:
            raise ValueError("a record of %d nodes does not fit in %d" % (len(record["nodes"]), nodes))
        if len(record["relationships"]) > edges:
            raise ValueError("a record of %d relationships does not fit in %d" % (len(record["relationships"]), edges))
        for at, node in enumerate(record["nodes"]):
            (x1, y1), (x2, y2) = node["points"]
            node_class[plan, at] = class_id[node["class"]]
            start_x[plan, at], start_y[plan, at] = x1 / side, y1 / side
            end_x[plan, at], end_y[plan, at] = x2 / side, y2 / side
        for at, related in enumerate(record["relationships"]):
            relationship[plan, at] = relationship_id[related["type"]]
            subject[plan, at], obj[plan, at] = related["nodes"]

    return node_class, start_x, start_y, end_x, end_y, relationship, subject, obj


def seeds(repo_id, part):
    """The seed of every plan of a part, in the order of its lines."""
    with open(_file(repo_id, "%s_labels.jsonl" % part), encoding="utf-8") as labels:
        return [json.loads(label)["seed"] for label in labels]


def _file(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
