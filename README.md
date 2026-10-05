# AnatomyLearn

An interactive 3D anatomy and radiology learning tool. The 3D model is built from a real CT scan, so every structure lines up with the imaging. Click an organ in 3D and the CT slices jump to it; click a structure on a slice and it lights up in 3D.

```
CT scan ──► TotalSegmentator ──► label map ──► one mesh per structure (body.glb)
   │                                  │                    │
   └──── NiiVue slice viewer ◄────────┴── structure id ────┴──► content/structures/*.md
```

## Layout

| Path | What |
|---|---|
| `pipeline/build.py` | CT → segmentation → meshes, `structures.json` registry and web volumes |
| `web/` | Vite + React app: React Three Fiber (3D) and NiiVue (CT slices) |
| `content/structures/` | Learning content: one Markdown file per structure (anatomy, physiology, pathophysiology, imaging) |

## Setup

Requires Python 3.10–3.13 and Node 20+.

```sh
# 1. Pipeline
cd pipeline
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Sample CT (abdomen/pelvis, 3 mm) from the TotalSegmentator repo
mkdir -p data/raw
curl -L -o data/raw/example_ct.nii.gz \
  https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files/example_ct.nii.gz

# Segment and export to web/public/data. The first run downloads model weights (about 135 MB).
.venv/bin/python build.py --ct data/raw/example_ct.nii.gz

# 2. Web app
cd ../web
npm install
npm run dev
```

### Pipeline options

- `--full`: the 1.5 mm TotalSegmentator model. More accurate, but slow without a GPU (the default is the 3 mm fast model).
- `--device mps|gpu`: run segmentation on Apple Silicon or an NVIDIA GPU.
- `--seg path.nii.gz`: reuse an existing multilabel segmentation.
- `--web-spacing 2`: resample large scans before shipping them to the browser.
- `--faces 8000`: triangle budget per structure.

For a whole-body model, run the pipeline on a whole-body CT. The TotalSegmentator training dataset on Zenodo (CC BY 4.0) includes many.

## Coordinates

Everything shares patient **RAS** millimetres from the CT's affine (x = right, y = anterior, z = superior). NiiVue uses RAS natively. Meshes are rotated into three.js Y-up space as `(-R, S, A)`, which is `RAS_TO_THREE` in `build.py` and `rasToThree` in `web/src/coords.ts`. Keep the two in sync.

## Adding content

Create `content/structures/<name>.md`:

```md
---
title: Kidney
ids: kidney_left, kidney_right
level: intro
reviewed: false
---
### Anatomy
...
```

`ids` are structure ids from `web/public/data/structures.json` (TotalSegmentator class names). Leave `reviewed: false` until a clinician has checked the page; the app shows a draft banner until then.

## Data sources and licenses

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator): Apache 2.0. Supplies the segmentation model, SNOMED CT codes and default colours. Some of its other tasks need a separate license; the `total` task used here does not.
- Sample CT: from the TotalSegmentator test data.
