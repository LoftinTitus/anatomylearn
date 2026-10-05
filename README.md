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

# Whole-body CT: the Visible Human Male (downloads about 130 MB, writes a 1.5 mm NIfTI)
.venv/bin/python sources/visible_human.py

# Segment (total + body outline) and export to web/public/data.
# The first run downloads model weights. --device mps uses the Apple Silicon GPU.
.venv/bin/python build.py --ct data/raw/visible_human_male_ct.nii.gz --device mps --web-spacing 2

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

- `--tasks total,body`: the TotalSegmentator tasks to run. `body` adds a translucent whole-body outline.

For a quick test on a small scan, use the TotalSegmentator sample CT (abdomen/pelvis):

```sh
curl -L -o data/raw/example_ct.nii.gz \
  https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files/example_ct.nii.gz
.venv/bin/python build.py --ct data/raw/example_ct.nii.gz
```

### About the Visible Human scan

[`sources/visible_human.py`](pipeline/sources/visible_human.py) converts NLM's fresh-cadaver CT (1993, GE Genesis format) into one volume. Things to know:

- **Missing stretch:** there is no CT from roughly the knee to the ankle. The app shows that stretch as empty and flags structures that run into it.
- **Arms cut off:** the arms lie partly outside the scanner's field of view, so they're cut off at the sides.
- **Cadaver artifacts:** expect postmortem gas in vessels and some streak artifact.
- **Bones not covered:** TotalSegmentator's free `total` task covers the skull down to the femurs and humeri. Forearm, hand, lower-leg and foot bones need the `appendicular_bones` task, which requires a license from the TotalSegmentator authors (free for non-commercial use).

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
- [Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html), National Library of Medicine. No license required since 2019; [NLM terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html) apply.
- Sample CT: from the TotalSegmentator test data.
