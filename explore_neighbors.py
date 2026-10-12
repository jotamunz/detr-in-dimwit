# %% What to look at
# The validation drawings of a d2s run on a map that turns and zooms, written as one HTML page. A
# drawing is the mean of its encoded tiles that hold ink, as `d2sEncodings` writes it, and UMAP
# places it in three dimensions. The map only says where a drawing is shown: clicking one marks its
# nearest neighbors by cosine similarity of the encodings themselves, so where they land on the
# map shows how far the map can be trusted around there.
import base64
import io
import json
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np
import umap
from huggingface_hub import hf_hub_download
from PIL import Image

RUNS = Path("results/runs")
RUN = RUNS / "vitruvion-final-d2po"

ENCODINGS = RUN / "d2s-encodings-vitruvion-primitives-s-deep-val-270000.npz"
DRAWINGS = hf_hub_download(repo_id="benikm91/sketch-graph-vitruvion", filename="records/val_images.npy", repo_type="dataset")

# What every drawing holds, from the transcripts scored beside the checkpoint.
TRANSCRIPTS = RUN / "d2s-vitruvion-primitives-s-deep-val-270000.jsonl"

# How many drawings the map holds. Their neighbors are looked for among them alone.
SAMPLE = 5000
NEIGHBORS = 8

# How many neighbors UMAP keeps a drawing beside: fewer favours fine detail, more the layout as a
# whole.
UMAP_NEIGHBORS = 15

# Which drawings are on the map, and where UMAP puts them: the same seed draws the same map.
SEED = 0


# %% The drawings, and what the encoder made of them
held = np.load(ENCODINGS)
embeddings = held["embeddings"]
drawings = np.load(DRAWINGS, mmap_mode="r")
assert len(embeddings) == len(drawings), "the encodings are not of this split"

# A drawing whose strokes are too small to leave ink has no embedding, and is left off the map.
inked = np.isfinite(embeddings).all(axis=1)
shown = np.sort(np.random.default_rng(SEED).choice(np.flatnonzero(inked), SAMPLE, replace=False))
directions = embeddings[shown] / np.linalg.norm(embeddings[shown], axis=1, keepdims=True)


# %% The nearest neighbors, in the space of the encodings
similarity = directions @ directions.T
np.fill_diagonal(similarity, -np.inf)  # a drawing is not its own neighbor
nearest = np.argsort(-similarity, axis=1)[:, :NEIGHBORS]
closeness = np.take_along_axis(similarity, nearest, axis=1)


# %% The map
placed = umap.UMAP(n_components=3, n_neighbors=UMAP_NEIGHBORS, metric="cosine", random_state=SEED).fit_transform(directions)


# %% What each drawing holds, and what it looks like
with open(TRANSCRIPTS, encoding="utf-8") as transcripts:
    lines = transcripts.readlines()


def classes_in(drawing):
    record = json.loads(lines[drawing])
    assert record["drawing"] == drawing, "the transcripts are not in the order of the split"
    return [node["class"] for node in record["target"]["nodes"]]


def described(classes):
    """`3 lines, 1 circle`, most frequent first."""
    held = Counter(classes).most_common()
    return ", ".join(f"{count} {name}{'' if count == 1 else 's'}" for name, count in held) or "nothing"


def png(drawing):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(drawings[drawing])).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


classes = [classes_in(drawing) for drawing in shown]
page = {
    "drawing": shown.tolist(),
    "x": np.round(placed[:, 0], 4).tolist(),
    "y": np.round(placed[:, 1], 4).tolist(),
    "z": np.round(placed[:, 2], 4).tolist(),
    "holds": [described(held) for held in classes],
    "picture": [png(drawing) for drawing in shown],
    "nearest": nearest.tolist(),
    "closeness": np.round(closeness, 4).tolist(),
}


# %% The page
# Every dot is a drawing. The selected drawing is the dark diamond; its nearest neighbors are the
# orange dots, each joined to it by a line.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Encoding explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  body { margin: 0; display: flex; height: 100vh; font: 14px system-ui, sans-serif; color: #222; background: #fff; }
  #map { flex: 1; min-width: 0; }
  aside { width: 340px; overflow-y: auto; padding: 16px; box-sizing: border-box; border-left: 1px solid #ddd; }
  h2 { font-size: 16px; margin: 0 0 8px; }
  h3 { font-size: 13px; margin: 16px 0 8px; color: #555; }
  img { display: block; border: 1px solid #ddd; }
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
  <div id="hovered">Hover over a dot to see its drawing.</div>
  <div id="selected">Click a dot to see its nearest neighbors.</div>
</aside>
<script>
const data = __DATA__;
const everyone = data.drawing.map((_, at) => at);
const label = at => `#${data.drawing[at]} · ${data.holds[at]}`;
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
    <p class="holds">${data.holds[at]}</p>
    <h3>Nearest by cosine similarity</h3>
    <ol id="neighbors">${near.map((neighbor, rank) => `
      <li data-at="${neighbor}">${picture(neighbor, 72)}
        <div><div>#${data.drawing[neighbor]} <span class="score">${data.closeness[at][rank].toFixed(3)}</span></div>
        <div class="holds">${data.holds[neighbor]}</div></div></li>`).join("")}
    </ol>`;
  document.querySelectorAll("#neighbors li").forEach(item => item.addEventListener("click", () => select(Number(item.dataset.at))));
}
</script>
</body>
</html>
"""

explorer = ENCODINGS.with_name(f"{ENCODINGS.stem}-explorer.html")
explorer.write_text(PAGE.replace("__DATA__", json.dumps(page)), encoding="utf-8")
print(f"{SAMPLE} drawings written to {explorer}")
webbrowser.open(explorer.resolve().as_uri())
