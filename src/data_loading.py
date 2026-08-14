import numpy as np
import nrrd
import pydicom

STRUCTURES = ["CLAV", "MSC", "VSC", "ASC", "K1", "PB"]
ALIAS = {"MCS": "MSC"}  # typo connue dans un fichier (2017-110_01-0223-V1MR/RSSFP D)
CANON = {"CLAV": 1, "MSC": 2, "VSC": 3, "ASC": 4, "K1": 5, "PB": 6}


def get_dicom_positions(series_dir):
    frames = []
    for f in sorted(series_dir.glob("IM_*.dcm")):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        frames.append({
            "instance_number": int(ds.InstanceNumber),
            "position": np.array(ds.ImagePositionPatient, dtype=float),
            "file": f,
        })
    frames.sort(key=lambda x: x["instance_number"])
    return frames


def build_frame_to_k(series_dir, seg_path, tol=1e-2):
    """Renvoie {instance_number: k} en recalculant la correspondance géométriquement
    entre l'axe des frames DICOM et l'axe des slices du .seg.nrrd (l'ordre stocké
    dans le nrrd n'est pas garanti identique à l'InstanceNumber DICOM).
    Une frame est 'unmatched' si aucune position DICOM ne correspond à la position
    attendue pour k à moins de `tol` mm (arrive quand IPP ne varie pas entre frames)."""
    frames = get_dicom_positions(series_dir)
    header = nrrd.read_header(str(seg_path))
    origin = np.array(header["space origin"], dtype=float)
    axis_dir = np.array(header["space directions"], dtype=float)[-1]
    n_k = header["sizes"][-1]

    mapping, unmatched = {}, []
    for k in range(n_k):
        pos_k = origin + k * axis_dir
        best = min(frames, key=lambda fr: np.linalg.norm(fr["position"] - pos_k))
        if np.linalg.norm(best["position"] - pos_k) > tol:
            unmatched.append(k)
        mapping[best["instance_number"]] = k
    return mapping, unmatched, frames


def load_canonical_labelmap(seg_path):
    """Charge un .seg.nrrd et renvoie un labelmap (H, W, K) avec des valeurs canoniques
    1..6 (voir CANON), résolues par nom de segment plutôt que par LabelValue brut.
    Les deux premiers axes du tableau nrrd sont transposés par rapport au pixel_array
    DICOM (i,j nrrd = j,i DICOM) — vérifié empiriquement par score de gradient de bord
    (voir notebooks/analyse_exploratoire_TOS.ipynb, section 5bis)."""
    data, header = nrrd.read(str(seg_path))
    is_4d = data.ndim == 4
    canonical = np.zeros(data.shape[-3:] if is_4d else data.shape, dtype=np.uint8)
    i = 0
    while f"Segment{i}_Name" in header:
        name = ALIAS.get(header[f"Segment{i}_Name"].strip().upper(), header[f"Segment{i}_Name"].strip().upper())
        label = int(header[f"Segment{i}_LabelValue"])
        layer = int(header.get(f"Segment{i}_Layer", 0))
        mask = (data[layer] == label) if is_4d else (data == label)
        canonical[mask] = CANON.get(name, 0)
        i += 1
    return canonical.transpose(1, 0, 2)


def load_aligned_pair(series_dir, seg_path):
    """Renvoie (image, label, status). image et label sont en ordre séquentiel
    InstanceNumber (index i -> InstanceNumber i+1), shape (H, W, n_frames).
    status vaut "GEOMETRIE_NON_RESOLUE" si l'axe des slices du nrrd ne peut pas être
    apparié sans ambiguïté aux frames DICOM (voir build_frame_to_k)."""
    mapping, unmatched, frames = build_frame_to_k(series_dir, seg_path)
    if unmatched:
        return None, None, "GEOMETRIE_NON_RESOLUE"

    canonical_k = load_canonical_labelmap(seg_path)
    n4_path = series_dir / "volume_n4.npy"
    if n4_path.exists():
        image = np.load(n4_path)
    else:
        image = np.stack([pydicom.dcmread(fr["file"]).pixel_array.astype(np.float32) for fr in frames], axis=-1)

    label = np.zeros_like(image, dtype=np.uint8)
    for fr in frames:
        label[:, :, fr["instance_number"] - 1] = canonical_k[:, :, mapping[fr["instance_number"]]]
    return image, label, "OK"
