# %% What to look at
# A sample of the validation drawings of a d2s run on a map that turns and zooms, written as one
# HTML page. How alike two drawings are is measured in one of three ways, as the command line says:
#   encoder  late interaction of their inked tiles, as the encoder's last block leaves them: every
#            tile's best cosine among the other drawing's tiles, averaged, both ways round
#   decoder  the nodes the decoder writes down for them, each as the prediction embedding it was
#            read from, paired one to one by the Hungarian algorithm: the best total cosine over the
#            larger number of nodes
#   pixels   late interaction of their inked tiles as drawn, the same tiles the encoder reads
# UMAP lays the drawings out in three dimensions by the same measure. The map only says where a
# drawing is shown: clicking one marks its nearest neighbors by the measure itself, so where they
# land on the map shows how far the map can be trusted around there.
# Run the whole file for the page: `python explore_neighbors.py decoder`.
import base64
import io
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np
import umap
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.optimize import linear_sum_assignment

RUNS = Path("results/runs")
RUN = RUNS / "vitruvion-final-d2po"

EMBEDDINGS = RUN / "d2s-embeddings-vitruvion-primitives-s-deep-val-270000.npz"
DRAWINGS = hf_hub_download(repo_id="benikm91/sketch-graph-vitruvion", filename="records/val_images.npy", repo_type="dataset")

# What every drawing holds, and what the model wrote down for it, from the transcripts scored
# beside the checkpoint.
TRANSCRIPTS = RUN / "d2s-vitruvion-primitives-s-deep-val-270000.jsonl"

SPACE = sys.argv[1] if len(sys.argv) > 1 else "encoder"

# How many pixels a side a tile is, as the encoder cuts a drawing into them.
TILE = 16

NEIGHBORS = 8

# How many neighbors UMAP keeps a drawing beside: fewer favours fine detail, more the layout as a
# whole.
UMAP_NEIGHBORS = 15

# Where UMAP puts the drawings: the same seed lays out the same map.
SEED = 0

# How many drawings have their tiles held against every inked tile at once, which is what bounds
# the memory of late interaction.
AT_ONCE = 50


# %% The drawings, and what the model made of them
saved = np.load(EMBEDDINGS)
drawings = np.load(DRAWINGS, mmap_mode="r")

# A drawing without ink, or one the decoder writes nothing down for, has nothing to be compared
# by. It is left off the map whatever the measure, so that every map holds the same drawings.
inked, written = saved["inked"] > 0, saved["written"] > 0
kept = inked.any(axis=1) & written.any(axis=1)
shown = saved["drawing"][kept]
patches, inked = saved["patches"][kept], inked[kept]
items, written = saved["items"][kept], written[kept]


# %% How alike every two drawings are
def normalized(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def tiles_of(pictures):
    """Every picture cut into tiles `TILE` pixels a side, each tile its ink, a pixel after another."""
    count, height, width = pictures.shape
    ink = 255 - pictures.astype(np.float32)  # blank is nothing, so that only ink is compared
    across, down = width // TILE, height // TILE
    return ink.reshape(count, down, TILE, across, TILE).transpose(0, 1, 3, 2, 4).reshape(count, down * across, TILE * TILE)


def late_interaction(tiles, inked):
    """How well every inked tile of one drawing finds a counterpart among the inked tiles of the
    other: the mean of each tile's best cosine there, averaged over both ways round."""
    held = normalized(tiles[inked])  # every inked tile, drawing after drawing
    counts = inked.sum(axis=1)
    starts = np.cumsum(counts) - counts

    def found(block):
        """For each drawing of the block, the mean best cosine of its tiles in every drawing."""
        own = held[starts[block[0]]:starts[block[-1]] + counts[block[-1]]]
        best = np.maximum.reduceat(own @ held.T, starts, axis=1)
        return np.add.reduceat(best, starts[block] - starts[block[0]], axis=0) / counts[block, None]

    blocks = np.array_split(np.arange(len(counts)), len(counts) // AT_ONCE + 1)
    one_way = np.concatenate([found(block) for block in blocks])
    return (one_way + one_way.T) / 2


def one_to_one(items, written):
    """How well the nodes written for one drawing pair up, one to one, with those written for the
    other: the best total cosine over every pairing, which the Hungarian algorithm finds, over the
    larger number of nodes, so that a node left over counts as nothing. A pair pointing apart counts
    as nothing too, as if left over."""
    nodes = normalized(items[written])
    counts = written.sum(axis=1)
    starts = np.cumsum(counts) - counts

    def paired_with_later(a):
        alike = np.maximum(nodes[starts[a]:starts[a] + counts[a]] @ nodes.T, 0)

        def paired(b):
            pairs = alike[:, starts[b]:starts[b] + counts[b]]
            return pairs[linear_sum_assignment(pairs, maximize=True)].sum() / max(counts[a], counts[b])

        return [paired(b) for b in range(a + 1, len(counts))]

    # Pairing is the same either way round, so every two drawings are paired once.
    later = np.zeros((len(counts), len(counts)))
    later[np.triu_indices(len(counts), 1)] = np.concatenate([paired_with_later(a) for a in range(len(counts))])
    return later + later.T + np.eye(len(counts))


def pixel_interaction():
    tiles = tiles_of(drawings[shown])
    return late_interaction(tiles, tiles.max(axis=-1) > 0)


# What each measure is, as the side panel names it, and how it is taken.
MEASURES = {
    "encoder": ("late interaction of the encoded tiles", lambda: late_interaction(patches, inked)),
    "decoder": ("one-to-one pairing of the written nodes", lambda: one_to_one(items, written)),
    "pixels": ("late interaction of the tiles as drawn", pixel_interaction),
}
assert SPACE in MEASURES, f"say which measure: {', '.join(MEASURES)}"
measure, taken = MEASURES[SPACE]
similarity = taken()


# %% The nearest neighbors, by the measure itself
# UMAP lays out distances, and a drawing is no distance from itself.
distances = np.clip(1 - similarity, 0, None)
np.fill_diagonal(distances, 0)

np.fill_diagonal(similarity, -np.inf)  # a drawing is not its own neighbor
nearest = np.argsort(-similarity, axis=1)[:, :NEIGHBORS]
closeness = np.take_along_axis(similarity, nearest, axis=1)


# %% The map
placed = umap.UMAP(n_components=3, n_neighbors=UMAP_NEIGHBORS, metric="precomputed", random_state=SEED).fit_transform(distances)


# %% What each drawing holds, and what it looks like
with open(TRANSCRIPTS, encoding="utf-8") as transcripts:
    lines = transcripts.readlines()


def record_of(drawing):
    record = json.loads(lines[drawing])
    assert record["drawing"] == drawing, "the transcripts are not in the order of the split"
    return record


def classes_in(record):
    return Counter(node["class"] for node in record["nodes"])


def described(classes):
    """`3 lines, 1 circle`, most frequent first."""
    ordered = sorted(classes.items(), key=lambda held: (-held[1], held[0]))
    return ", ".join(f"{count} {name}{'' if count == 1 else 's'}" for name, count in ordered) or "nothing"


def png(drawing):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(drawings[drawing])).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


records = [record_of(drawing) for drawing in shown]
holds = [classes_in(record["target"]) for record in records]
page = {
    "measure": measure,
    "drawing": shown.tolist(),
    "x": np.round(placed[:, 0], 4).tolist(),
    "y": np.round(placed[:, 1], 4).tolist(),
    "z": np.round(placed[:, 2], 4).tolist(),
    "holds": [described(held) for held in holds],
    "picture": [png(drawing) for drawing in shown],
    "nearest": nearest.tolist(),
    "closeness": np.round(closeness, 4).tolist(),
}

# The decoder is measured by what the model wrote down, so its page shows that beside what the
# drawing holds.
if SPACE == "decoder":
    page["wrote"] = [described(classes_in(record["predicted"])) for record in records]


# %% The page
# Every dot is a drawing. The selected drawing is the dark diamond; its nearest neighbors are the
# orange dots, each joined to it by a line.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__SPACE__ explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  body { margin: 0; display: flex; height: 100vh; font: 14px system-ui, sans-serif; color: #222; background: #fff; }
  #map { flex: 1; min-width: 0; }
  aside { width: 340px; overflow-y: auto; padding: 16px; box-sizing: border-box; border-left: 1px solid #ddd; }
  h2 { font-size: 16px; margin: 0 0 8px; }
  h3 { font-size: 13px; margin: 16px 0 8px; color: #555; }
  img { display: block; border: 1px solid #ddd; }
  #search { display: flex; gap: 6px; }
  #search input { flex: 1; min-width: 0; padding: 4px 6px; font: inherit; }
  #found { min-height: 1em; margin: 4px 0 12px; }
  #hovered { display: flex; gap: 8px; align-items: center; min-height: 66px; margin-bottom: 16px; color: #777; }
  #neighbors { list-style: none; margin: 0; padding: 0; }
  #neighbors li { display: flex; gap: 10px; align-items: center; padding: 4px; border-radius: 4px; cursor: pointer; }
  #neighbors li:hover { background: #f2f2f2; }
  .score { font-weight: 600; font-variant-numeric: tabular-nums; }
  .holds { color: #666; font-size: 12px; }
</style>
</head>
<body>
<div id="map"></div>
<aside>
  <form id="search"><input id="wanted" type="search" placeholder="Find a drawing by its number"><button>Find</button></form>
  <p id="found" class="holds"></p>
  <div id="hovered">Hover over a dot to see its drawing.</div>
  <div id="selected">Click a dot to see its nearest neighbors.</div>
</aside>
<script>
const data = __DATA__;
const everyone = data.drawing.map((_, at) => at);
const contents = at => data.wrote
  ? `true: ${data.holds[at]}<br>written: ${data.wrote[at]}`
  : data.holds[at];
const label = at => `#${data.drawing[at]} · ${contents(at)}`;
const picture = (at, size) => `<img src="${data.picture[at]}" width="${size}" height="${size}" alt="drawing ${data.drawing[at]}">`;
const along = (axis, ats) => ats.map(at => data[axis][at]);

const map = document.getElementById("map");
const dots = { type: "scatter3d", mode: "markers", hovertemplate: "%{text}<extra></extra>" };
Plotly.newPlot(map, [
  { ...dots, x: data.x, y: data.y, z: data.z, customdata: everyone, text: everyone.map(label),
    marker: { size: 2.5, opacity: 0.8, color: "#4a7bb7" } },
  { type: "scatter3d", mode: "lines", x: [], y: [], z: [], hoverinfo: "skip", line: { color: "#e4572e", width: 3 } },
  { ...dots, x: [], y: [], z: [], customdata: [], text: [], marker: { size: 6, color: "#e4572e", line: { color: "#fff", width: 1 } } },
  { ...dots, x: [], y: [], z: [], customdata: [], text: [], marker: { size: 9, color: "#111", symbol: "diamond" } },
], {
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  scene: Object.fromEntries(["xaxis", "yaxis", "zaxis"].map(axis => [axis, { showticklabels: false, title: { text: "" } }])),
}, { responsive: true, displaylogo: false });

map.on("plotly_hover", event => {
  const at = event.points[0].customdata;
  document.getElementById("hovered").innerHTML = `${picture(at, 64)}<span>${label(at)}</span>`;
});

// Plotly emits a 3D click while it draws, for as long as the button is held, and selecting draws
// again. So a selection waits for the drawing to finish, and the drawing already selected is not
// selected again.
let selected = null;
map.on("plotly_click", event => {
  const at = event.points[0].customdata;
  if (at !== selected) setTimeout(() => select(at));
});

const onMap = new Map(data.drawing.map((drawing, at) => [drawing, at]));
document.getElementById("search").addEventListener("submit", event => {
  event.preventDefault();
  const typed = document.getElementById("wanted").value.trim().replace(/^#/, "");
  const at = onMap.get(Number(typed));
  const found = document.getElementById("found");
  if (typed === "" || !Number.isInteger(Number(typed))) found.textContent = "A drawing is found by its number, such as 1834.";
  else if (at === undefined) found.textContent = `#${typed} is not one of the ${data.drawing.length} drawings on the map.`;
  else {
    found.textContent = "";
    select(at);
  }
});

function select(at) {
  selected = at;
  const near = data.nearest[at];
  const links = axis => near.flatMap(neighbor => [data[axis][at], data[axis][neighbor], null]);
  Plotly.restyle(map, { x: [links("x")], y: [links("y")], z: [links("z")] }, [1]);
  Plotly.restyle(map, { x: [along("x", near)], y: [along("y", near)], z: [along("z", near)], customdata: [near], text: [near.map(label)] }, [2]);
  Plotly.restyle(map, { x: [[data.x[at]]], y: [[data.y[at]]], z: [[data.z[at]]], customdata: [[at]], text: [[label(at)]] }, [3]);
  document.getElementById("selected").innerHTML = `
    <h2>#${data.drawing[at]}</h2>
    ${picture(at, 192)}
    <p class="holds">${contents(at)}</p>
    <h3>Nearest by ${data.measure}</h3>
    <ol id="neighbors">${near.map((neighbor, rank) => `
      <li data-at="${neighbor}">${picture(neighbor, 72)}
        <div><div>#${data.drawing[neighbor]} <span class="score">${data.closeness[at][rank].toFixed(3)}</span></div>
        <div class="holds">${contents(neighbor)}</div></div></li>`).join("")}
    </ol>`;
  document.querySelectorAll("#neighbors li").forEach(item => item.addEventListener("click", () => select(Number(item.dataset.at))));
}
</script>
</body>
</html>
"""

explorer = EMBEDDINGS.with_name(f"{EMBEDDINGS.stem}-{SPACE}-explorer.html")
explorer.write_text(PAGE.replace("__SPACE__", SPACE.capitalize()).replace("__DATA__", json.dumps(page)), encoding="utf-8")
print(f"{len(shown)} drawings written to {explorer}")
webbrowser.open(explorer.resolve().as_uri())
