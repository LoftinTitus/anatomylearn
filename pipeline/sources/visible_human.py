"""Download the Visible Human Male fresh-cadaver CT and convert it to one NIfTI volume.

Source: NLM Visible Human Project (no license required since 2019; NLM terms apply).
https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/radiological/normalCT/

The scan is 521 GE Genesis slices in two series with varying slice spacing
(1 mm head, 3 mm trunk, 5 mm legs) and varying field of view. Series 5 (legs)
was scanned feet-first with its own table coordinates, so the vertical position
comes from the file number instead: c_vmNNNN matches cryosection NNNN, which is
continuous head-to-toe at 1 mm. In-plane position comes from each slice header.

There is no CT for cryosections 2264-2687 (knee to ankle); that stretch is
written as air and listed under "noData" in the sidecar JSON.

    python sources/visible_human.py            # -> data/raw/visible_human_male_ct.nii.gz
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BASE = "https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/radiological"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "visible_human_male"
OUT = ROOT / "data" / "raw" / "visible_human_male_ct.nii.gz"

HEADER_BYTES = 3416  # IMGF control header length (constant for this series)
HU_OFFSET = -1024
# Cryosection number -> patient superior coordinate, chosen so series 2 matches its scanner locations.
Z_ORIGIN = 1402
MAX_INTERP_GAP_MM = 10  # wider gaps between slices are left empty rather than interpolated


def list_files(folder: str) -> list[str]:
    html = urllib.request.urlopen(f"{BASE}/{folder}/index.html").read().decode()
    return sorted(set(re.findall(r"href=['\"](c_vm\d+\.fre(?:\.Z|\.txt))['\"]", html)))


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def download() -> list[int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    images = list_files("normalCT")
    headers = list_files("normalCTHeaders")
    jobs = [(f"{BASE}/normalCT/{f}", CACHE / f) for f in images]
    jobs += [(f"{BASE}/normalCTHeaders/{f}", CACHE / f) for f in headers]
    print(f"Downloading {len(images)} slices + {len(headers)} headers to {CACHE} ...")
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda j: fetch(*j), jobs))
    # Use only slices that have both pixel data and a header.
    numbers = {int(re.search(r"\d+", f).group()) for f in images}
    numbers &= {int(re.search(r"\d+", f).group()) for f in headers}
    return sorted(numbers)


def read_header(n: int) -> dict[str, float]:
    text = (CACHE / f"c_vm{n}.fre.txt").read_bytes().replace(b"\0", b"").decode("latin-1")

    def field(name: str) -> float:
        m = re.search(rf"^{re.escape(name)}\.*: *(-?[\d.]+)", text, re.M)
        if not m:
            raise ValueError(f"c_vm{n}: missing '{name}'")
        return float(m.group(1))

    return {
        "px": field("Image pixel size - X"),
        "py": field("Image pixel size - Y"),
        "r": field("Center R coord of plane image"),
        "a": field("Center A coord of plane image"),
    }


def read_pixels(n: int) -> np.ndarray:
    raw = subprocess.run(["gzip", "-dc", str(CACHE / f"c_vm{n}.fre.Z")], check=True, capture_output=True).stdout
    if raw[:4] != b"IMGF":
        raise ValueError(f"c_vm{n}: not a GE Genesis file")
    # Rows run anterior -> posterior, columns patient right -> left.
    return np.frombuffer(raw[HEADER_BYTES:], ">u2").reshape(512, 512).astype(np.float32) + HU_OFFSET


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, default=1.5, help="Output isotropic voxel size in mm")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    s = args.spacing

    numbers = download()
    headers = {n: read_header(n) for n in numbers}

    # Common in-plane grid covering every slice's field of view.
    r_lo = min(h["r"] - 256 * h["px"] for h in headers.values())
    r_hi = max(h["r"] + 256 * h["px"] for h in headers.values())
    a_lo = min(h["a"] - 256 * h["py"] for h in headers.values())
    a_hi = max(h["a"] + 256 * h["py"] for h in headers.values())
    xs = np.arange(r_lo, r_hi, s)  # RAS x (patient right = +)
    ys = np.arange(a_lo, a_hi, s)  # RAS y (anterior = +)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    print(f"Resampling {len(numbers)} slices onto a {len(xs)}x{len(ys)} grid at {s} mm ...")
    planes = {}
    for n in numbers:
        h = headers[n]
        col = 255.5 - (gx - h["r"]) / h["px"]
        row = 255.5 - (gy - h["a"]) / h["py"]
        planes[n] = ndimage.map_coordinates(read_pixels(n), [row, col], order=1, cval=HU_OFFSET)

    # Vertical axis: z = Z_ORIGIN - cryosection number (superior = +).
    src_z = np.array([Z_ORIGIN - n for n in numbers], dtype=np.float64)  # descending
    order = np.argsort(src_z)
    src_z, src_n = src_z[order], [numbers[i] for i in order]
    zs = np.arange(src_z[0], src_z[-1] + 1e-6, s)

    vol = np.full((len(xs), len(ys), len(zs)), HU_OFFSET, dtype=np.int16)
    for k, z in enumerate(zs):
        i = int(np.searchsorted(src_z, z, side="right"))
        lo, hi = max(i - 1, 0), min(i, len(src_z) - 1)
        if lo == hi or src_z[hi] - src_z[lo] < 1e-6:
            vol[:, :, k] = planes[src_n[lo]]
            continue
        gap = src_z[hi] - src_z[lo]
        if gap > MAX_INTERP_GAP_MM:
            continue
        t = (z - src_z[lo]) / gap
        vol[:, :, k] = np.round((1 - t) * planes[src_n[lo]] + t * planes[src_n[hi]])

    no_data = [
        [round(float(src_z[i]), 1), round(float(src_z[i + 1]), 1)]
        for i in range(len(src_z) - 1)
        if src_z[i + 1] - src_z[i] > MAX_INTERP_GAP_MM
    ]
    affine = np.diag([s, s, s, 1.0])
    affine[:3, 3] = [xs[0], ys[0], zs[0]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, affine), args.out)
    sidecar = {
        "source": "Visible Human Male, fresh-cadaver CT (NLM Visible Human Project)",
        "sourceUrl": f"{BASE}/normalCT/",
        "terms": "https://www.nlm.nih.gov/databases/download/terms_and_conditions.html",
        "noData": [{"zRange": r, "note": "No CT acquired for this stretch (approx. knee to ankle)"} for r in no_data],
    }
    args.out.with_name(args.out.name.replace(".nii.gz", ".json")).write_text(json.dumps(sidecar, indent=1))
    print(f"Wrote {args.out} {vol.shape} ({args.out.stat().st_size / 1e6:.0f} MB); no-data z ranges: {no_data}")


if __name__ == "__main__":
    main()
