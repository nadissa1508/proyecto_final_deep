"""Lectura de RoCoLe y BRACOL y mapeo de sus etiquetas a las 4 clases del proyecto.

Clases (PLAN.md, D2), por % del área foliar afectada:

    0 sana · 1 roya leve (≤ 5 %) · 2 roya severa (> 5 %) · 3 otra afección

Este es el **único** lugar donde se define el mapeo. Todo lo demás (splits,
entrenamiento, métricas) consume el manifiesto que genera `build_manifest`.

RoCoLe (`RoCoLe-classes.xlsx`, columna `Multiclass.Label`, escala OIRSA):
    healthy → 0 · rust_level_1 (1–5 %) → 1 · rust_level_2..4 (> 5 %) → 2
    red_spider_mite → 3
    Grupo para el split: la planta, tomada del nombre `C<c>P<p><E|H><n>.jpg`
    (E = envés, H = haz). Cada planta tiene 4 fotos de hojas distintas.

BRACOL (`leaf/dataset.csv`; leyenda en `leaf/legend.txt`):
    `predominant_stress`: 0 sana, 1 minador, 2 roya, 3 phoma (mancha marrón),
    4 cercospora, 5 indeterminado. `severity` 0..4 (sana, muy baja 0.1–5 %,
    baja 5–10 %, alta 10–15 %, muy alta > 15 %) se refiere al estrés
    predominante. Columnas booleanas `miner, rust, phoma, cercospora` indican
    presencia de cada estrés.
    sana → 0 · roya predominante con severidad 1 → 1 · roya predominante con
    severidad 2..4 → 2 · minador/phoma/cercospora predominante **sin** roya → 3.
    Se excluyen (con motivo en el manifiesto):
      - `predominant_stress == 5` (indeterminado);
      - hojas con roya presente pero no predominante: tienen roya, pero su
        severidad no es de la roya, así que no se puede asignar 1 ni 2, y
        etiquetarlas como "otra afección" enseñaría que esa roya no es roya.
    Grupo para el split: el id de la imagen (no hay identificador de planta).

Duplicados (`resolve_duplicates`), detectados con SHA-256 (copia exacta) y
pHash (distancia de Hamming ≤ 4 sobre 64 bits = casi idéntica; solo en RoCoLe,
ver `NEAR_DUPLICATE_DATASETS`). RoCoLe trae la
misma foto bajo dos nombres de planta en 67 casos, a veces con etiquetas
distintas. Reglas:
    - imágenes duplicadas o casi duplicadas se unen en un mismo grupo, para que
      nunca queden separadas entre train y test;
    - si sus clases del proyecto no coinciden, se excluyen todas (no se puede
      saber cuál etiqueta es la correcta);
    - si coinciden, se conserva una sola copia de cada duplicado exacto.

Uso:
    python -m src.vision.datasets      # escribe data/processed/vision/manifest.csv e imprime resumen
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from src.common.paths import PROCESSED_DIR, RAW_DIR

CLASSES = ("sana", "roya_leve", "roya_severa", "otra_afeccion")

ROCOLE_DIR = RAW_DIR / "rocole"
BRACOL_DIR = RAW_DIR / "bracol" / "extracted" / "coffee-datasets" / "leaf"
MANIFEST_PATH = PROCESSED_DIR / "vision" / "manifest.csv"
HASHES_PATH = PROCESSED_DIR / "vision" / "hashes.csv"
NEAR_DUPLICATE_DISTANCE = 4
# El pHash solo se usa en RoCoLe (fondo de campo, muy distintivo). En BRACOL
# todas las hojas son alargadas sobre fondo blanco y el pHash marca como
# "casi idénticas" hojas distintas (verificado visualmente); ahí solo cuentan
# las copias exactas por SHA-256.
NEAR_DUPLICATE_DATASETS = ("rocole",)

ROCOLE_MAP = {
    "healthy": 0,
    "rust_level_1": 1,
    "rust_level_2": 2,
    "rust_level_3": 2,
    "rust_level_4": 2,
    "red_spider_mite": 3,
}
_ROCOLE_NAME = re.compile(r"^C(\d+)P(\d+)")

BRACOL_STRESS = {0: "healthy", 1: "miner", 2: "rust", 3: "phoma", 4: "cercospora", 5: "undetermined"}


def load_rocole() -> pd.DataFrame:
    df = pd.read_excel(ROCOLE_DIR / "RoCoLe-classes.xlsx")
    ids = df["File"].str.extract(_ROCOLE_NAME)
    if ids.isna().any().any():
        raise ValueError("Archivos de RoCoLe sin identificador de planta en el nombre")
    return pd.DataFrame({
        "path": [str(ROCOLE_DIR / "images" / f) for f in df["File"]],
        "dataset": "rocole",
        "group": "rocole_C" + ids[0] + "P" + ids[1],
        "original_label": df["Multiclass.Label"],
        "severity": pd.NA,
        "label": df["Multiclass.Label"].map(ROCOLE_MAP),
        "excluded": "",
    })


def _bracol_label(row: pd.Series) -> tuple:
    stress = int(row["predominant_stress"])
    if stress == 5:
        return pd.NA, "estres_indeterminado"
    if stress == 0:
        return 0, ""
    if stress == 2:
        return (1 if int(row["severity"]) == 1 else 2), ""
    if int(row["rust"]) == 1:
        return pd.NA, "roya_no_predominante"
    return 3, ""


def load_bracol() -> pd.DataFrame:
    df = pd.read_csv(BRACOL_DIR / "dataset.csv")
    labels = df.apply(_bracol_label, axis=1, result_type="expand")
    return pd.DataFrame({
        "path": [str(BRACOL_DIR / "images" / f"{i}.jpg") for i in df["id"]],
        "dataset": "bracol",
        "group": "bracol_" + df["id"].astype(str),
        "original_label": df["predominant_stress"].map(BRACOL_STRESS),
        "severity": df["severity"],
        "label": labels[0],
        "excluded": labels[1],
    })


def image_hashes(paths: pd.Series) -> pd.DataFrame:
    """SHA-256 y pHash de cada imagen, con caché en `HASHES_PATH`."""
    import imagehash
    from PIL import Image

    cache = pd.read_csv(HASHES_PATH) if HASHES_PATH.exists() else pd.DataFrame(columns=["path", "sha256", "phash"])
    known = dict(zip(cache["path"], zip(cache["sha256"], cache["phash"])))
    rows = []
    for path in paths:
        if path not in known:
            with open(path, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            with Image.open(path) as im:
                im.draft("RGB", (512, 512))
                known[path] = (sha, str(imagehash.phash(im.convert("RGB"))))
        rows.append((path, *known[path]))
    out = pd.DataFrame(rows, columns=["path", "sha256", "phash"])
    HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(HASHES_PATH, index=False)
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)

    def union_all(self, idx: list) -> None:
        for k in idx[1:]:
            self.union(idx[0], k)


def resolve_duplicates(manifest: pd.DataFrame) -> pd.DataFrame:
    import imagehash

    m = manifest.reset_index(drop=True).copy()
    hashes = image_hashes(m["path"])
    m["sha256"], m["phash"] = hashes["sha256"].values, hashes["phash"].values

    # Dos uniones independientes: `split_uf` une planta + duplicados (define el
    # grupo del split); `dup_uf` une solo imágenes duplicadas entre sí (define
    # qué etiquetas deben coincidir).
    split_uf, dup_uf = _UnionFind(len(m)), _UnionFind(len(m))
    for _, idx in m.groupby("group").groups.items():
        split_uf.union_all(list(idx))
    for _, idx in m.groupby("sha256").groups.items():
        split_uf.union_all(list(idx))
        dup_uf.union_all(list(idx))
    ph = [imagehash.hex_to_hash(h) for h in m["phash"]]
    near_ok = list(m["dataset"].isin(NEAR_DUPLICATE_DATASETS))
    for i in range(len(ph)):
        if not near_ok[i]:
            continue
        for j in range(i + 1, len(ph)):
            if near_ok[j] and ph[i] - ph[j] <= NEAR_DUPLICATE_DISTANCE:
                split_uf.union(i, j)
                dup_uf.union(i, j)

    m["group"] = [m.at[split_uf.find(i), "group"] for i in range(len(m))]
    m["dup_cluster"] = [dup_uf.find(i) for i in range(len(m))]
    sizes = m["dup_cluster"].map(m["dup_cluster"].value_counts())
    active = (m["excluded"] == "") & (sizes > 1)
    conflict = m[active].groupby("dup_cluster")["label"].transform("nunique") > 1
    m.loc[conflict[conflict].index, "excluded"] = "duplicado_etiqueta_contradictoria"
    keep_first = m[m["excluded"] == ""].duplicated("sha256", keep="first")
    m.loc[keep_first[keep_first].index, "excluded"] = "copia_exacta"
    return m


def build_manifest() -> pd.DataFrame:
    manifest = pd.concat([load_rocole(), load_bracol()], ignore_index=True)
    if manifest.loc[manifest["excluded"] == "", "label"].isna().any():
        raise ValueError("Hay etiquetas sin mapear: revisar ROCOLE_MAP / _bracol_label")
    missing = [p for p in manifest["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imágenes del manifiesto no existen, p. ej. {missing[0]}")
    manifest["label"] = manifest["label"].astype("Int64")
    return resolve_duplicates(manifest)


def summary(manifest: pd.DataFrame) -> pd.DataFrame:
    kept = manifest[manifest["excluded"] == ""]
    table = pd.crosstab(kept["label"].map(dict(enumerate(CLASSES))), kept["dataset"], margins=True)
    return table


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifiesto: {MANIFEST_PATH} ({len(manifest)} imágenes)")
    print("\nImágenes por clase y dataset (sin excluidas):")
    print(summary(manifest))
    print("\nExcluidas:")
    print(manifest.loc[manifest["excluded"] != "", ["dataset", "excluded"]].value_counts())
    print("\nGrupos para el split:", manifest.groupby("dataset")["group"].nunique().to_dict())


if __name__ == "__main__":
    main()
