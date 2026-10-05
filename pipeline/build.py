"""Build web assets from a CT scan.

    CT (NIfTI) -> TotalSegmentator label map -> one mesh per structure (GLB)
                                             -> structures.json registry
                                             -> ct.nii.gz + seg.nii.gz for the slice viewer

Every output shares one coordinate space: patient RAS millimetres from the CT
affine. The slice viewer (NiiVue) works in RAS mm natively; meshes are rotated
into three.js's Y-up frame by RAS_TO_THREE, which the web app inverts.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import fast_simplification
import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "web" / "public" / "data"

# RAS (x=right, y=anterior, z=superior) -> three.js (x, y=up, z=toward viewer).
# three.x = patient's left, three.y = superior, three.z = anterior, so a camera
# on +z sees the patient in anatomical position (patient's right on screen left).
# Proper rotation (det = +1), so triangle winding is preserved.
RAS_TO_THREE = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)

# Ordered: first matching prefix wins.
SYSTEM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("urinary", ("kidney_cyst",)),  # pathology, flagged separately
    ("skeletal", ("vertebrae_", "rib_", "sacrum", "humerus", "scapula", "clavicula",
                  "femur", "hip_", "skull", "sternum", "costal_cartilages")),
    ("muscular", ("gluteus_", "autochthon", "iliopsoas")),
    ("cardiovascular", ("heart", "aorta", "pulmonary_vein", "pulmonary_artery",
                        "brachiocephalic", "subclavian", "common_carotid", "atrial_appendage",
                        "superior_vena_cava", "inferior_vena_cava", "portal_vein",
                        "iliac_artery", "iliac_vena")),
    ("respiratory", ("lung_", "trachea")),
    ("digestive", ("esophagus", "stomach", "small_bowel", "duodenum", "colon", "liver",
                   "gallbladder", "pancreas")),
    ("urinary", ("kidney", "urinary_bladder")),
    ("reproductive", ("prostate",)),
    ("endocrine", ("thyroid", "adrenal")),
    ("lymphatic", ("spleen",)),
    ("nervous", ("brain", "spinal_cord")),
]

PATHOLOGY = {"kidney_cyst_left", "kidney_cyst_right"}

# Display names where "<Side> <SNOMED meaning>" reads badly.
NAME_OVERRIDES = {
    "autochthon_left": "Left erector spinae (deep back muscles)",
    "autochthon_right": "Right erector spinae (deep back muscles)",
    "hip_left": "Left hip bone",
    "hip_right": "Right hip bone",
    "gluteus_minimus_left": "Left gluteus minimus muscle",
    "gluteus_minimus_right": "Right gluteus minimus muscle",
    "kidney_cyst_left": "Left kidney cyst",
    "kidney_cyst_right": "Right kidney cyst",
    "atrial_appendage_left": "Left atrial appendage",
    "brachiocephalic_trunk": "Brachiocephalic trunk",
    "small_bowel": "Small intestine",
}


def system_for(slug: str) -> str:
    for system, prefixes in SYSTEM_RULES:
        if slug.startswith(prefixes):
            return system
    return "other"


def load_snomed() -> dict[str, dict]:
    import totalsegmentator

    path = Path(totalsegmentator.__file__).parent / "resources" / "totalsegmentator_snomed_mapping.csv"
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["Structure"]] = {
                "code": row["SegmentedPropertyTypeCodeSequence.CodeValue"],
                "meaning": row["SegmentedPropertyTypeCodeSequence.CodeMeaning"],
                "modifier": row["SegmentedPropertyTypeModifierCodeSequence.CodeMeaning"] or None,
                "color": [int(c) for c in row["DicomRGBColor"].split(",")],
            }
    return out


def display_name(slug: str, snomed: dict | None) -> str:
    if slug in NAME_OVERRIDES:
        return NAME_OVERRIDES[slug]
    if not snomed:
        return slug.replace("_", " ").capitalize()
    meaning = snomed["meaning"]
    if snomed["modifier"] in ("Left", "Right"):
        return f"{snomed['modifier']} {meaning[0].lower()}{meaning[1:]}"
    return meaning


def segment(ct_path: Path, seg_path: Path, fast: bool, device: str) -> None:
    from totalsegmentator.python_api import totalsegmentator

    print(f"Segmenting {ct_path.name} (fast={fast}, device={device}) ...")
    t = time.time()
    totalsegmentator(ct_path, seg_path, ml=True, fast=fast, device=device, quiet=True)
    print(f"  done in {time.time() - t:.0f}s")


def resample(img: nib.Nifti1Image, spacing: float, order: int) -> nib.Nifti1Image:
    """Resample to isotropic `spacing` mm (order 0 for label maps)."""
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    factors = zooms / spacing
    if np.allclose(factors, 1):
        return img
    data = ndimage.zoom(np.asanyarray(img.dataobj), factors, order=order)
    affine = img.affine.copy()
    affine[:3, :3] = img.affine[:3, :3] @ np.diag(1 / factors)
    return nib.Nifti1Image(data, affine)


def interior_point(mask: np.ndarray) -> np.ndarray:
    """Voxel index deepest inside the mask -- always inside, unlike the centroid
    of a curved structure (colon, ribs)."""
    dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
    return np.array(np.unravel_index(np.argmax(dist), mask.shape), dtype=np.float64)


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray, target_faces: int) -> trimesh.Trimesh | None:
    # Light blur before marching cubes removes the voxel staircase.
    field = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma=0.8)
    if field.max() < 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5)
    verts -= 2  # undo padding
    if len(faces) > target_faces:
        verts, faces = fast_simplification.simplify(
            verts.astype(np.float32), faces, target_reduction=1 - target_faces / len(faces))
    ras = nib.affines.apply_affine(affine, verts)
    mesh = trimesh.Trimesh(ras @ RAS_TO_THREE.T, faces, process=True)
    # Keep normals pointing outward regardless of affine handedness.
    if mesh.volume < 0:
        mesh.invert()
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", type=Path, required=True, help="CT volume (.nii / .nii.gz)")
    ap.add_argument("--seg", type=Path, help="Existing TotalSegmentator multilabel output; skips segmentation")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--full", action="store_true", help="Full-resolution 1.5mm model (slow on CPU); default is --fast 3mm")
    ap.add_argument("--device", default="cpu", help="cpu | gpu | mps")
    ap.add_argument("--web-spacing", type=float, default=0.0,
                    help="Resample volumes for the web to this isotropic spacing in mm (0 = keep)")
    ap.add_argument("--faces", type=int, default=8000, help="Target triangles per structure")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "data" / "work"
    work.mkdir(parents=True, exist_ok=True)

    seg_path = args.seg
    if seg_path is None:
        seg_path = work / f"{args.ct.name.split('.')[0]}_seg.nii.gz"
        if not seg_path.exists():
            segment(args.ct, seg_path, fast=not args.full, device=args.device)
        else:
            print(f"Reusing {seg_path}")

    # Canonical RAS orientation so voxel axes match world axes in every consumer.
    ct = nib.as_closest_canonical(nib.load(args.ct))
    seg = nib.as_closest_canonical(nib.load(seg_path))
    if ct.shape != seg.shape or not np.allclose(ct.affine, seg.affine, atol=1e-3):
        raise SystemExit(f"CT {ct.shape} and segmentation {seg.shape} are not on the same grid")

    if args.web_spacing:
        ct = resample(ct, args.web_spacing, order=1)
        seg = resample(seg, args.web_spacing, order=0)

    ct_data = np.clip(np.asanyarray(ct.dataobj), -1024, 3071).astype(np.int16)
    seg_data = np.asanyarray(seg.dataobj).astype(np.uint8)
    affine = seg.affine
    nib.save(nib.Nifti1Image(ct_data, ct.affine), args.out / "ct.nii.gz")
    nib.save(nib.Nifti1Image(seg_data, affine), args.out / "seg.nii.gz")

    from totalsegmentator.map_to_binary import class_map

    labels = class_map["total"]
    snomed = load_snomed()
    voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000

    scene = trimesh.Scene()
    structures = []
    objects = ndimage.find_objects(seg_data)
    for label, slices in enumerate(objects, start=1):
        if slices is None or label not in labels:
            continue
        slug = labels[label]
        # Crop to the bounding box (+margin) so per-structure work is cheap.
        lo = [max(s.start - 3, 0) for s in slices]
        hi = [min(s.stop + 3, n) for s, n in zip(slices, seg_data.shape)]
        crop = seg_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == label
        voxels = int(crop.sum())
        if voxels < 10:
            continue
        crop_affine = affine.copy()
        crop_affine[:3, 3] = nib.affines.apply_affine(affine, lo)

        mesh = mask_to_mesh(crop, crop_affine, args.faces)
        if mesh is None or len(mesh.faces) == 0:
            continue
        sn = snomed.get(slug)
        color = sn["color"] if sn else [200, 200, 200]
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(name=slug, baseColorFactor=[*color, 255]))
        scene.add_geometry(mesh, node_name=slug, geom_name=slug)

        bbox_vox = np.array([lo, hi], dtype=np.float64)
        bbox_ras = nib.affines.apply_affine(affine, bbox_vox)
        structures.append({
            "id": slug,
            "label": label,
            "name": display_name(slug, sn),
            "system": system_for(slug),
            "pathology": slug in PATHOLOGY,
            "color": color,
            "snomed": {"code": sn["code"], "meaning": sn["meaning"], "laterality": sn["modifier"]} if sn else None,
            "anchorRas": nib.affines.apply_affine(crop_affine, interior_point(crop)).round(1).tolist(),
            "bboxRas": [bbox_ras.min(0).round(1).tolist(), bbox_ras.max(0).round(1).tolist()],
            "volumeMl": round(voxels * voxel_ml, 1),
            # Touches the scan edge -> structure is cut off, mesh is partial.
            "truncated": any(a == 0 or b == n for a, b, n in
                             zip((s.start for s in slices), (s.stop for s in slices), seg_data.shape)),
        })
        print(f"  {label:3d} {slug:32s} {len(mesh.faces):6d} faces")

    scene.export(args.out / "body.glb")
    corners = nib.affines.apply_affine(affine, np.array([[0, 0, 0], np.array(seg_data.shape) - 1]))
    manifest = {
        "source": args.ct.name,
        "segmentation": "TotalSegmentator (total task, " + ("1.5mm" if args.full else "3mm fast") + ")",
        "coordinateSystem": "RAS mm; meshes pre-rotated by rasToThree",
        "rasToThree": RAS_TO_THREE.tolist(),
        "volumeBoundsRas": [corners.min(0).round(1).tolist(), corners.max(0).round(1).tolist()],
        "structures": structures,
    }
    (args.out / "structures.json").write_text(json.dumps(manifest, indent=1))
    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f}MB" for p in args.out.iterdir() if p.is_file()}
    print(f"Wrote {len(structures)} structures to {args.out}: {sizes}")


if __name__ == "__main__":
    main()
