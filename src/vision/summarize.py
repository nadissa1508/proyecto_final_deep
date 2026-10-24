"""Resume todas las corridas de visión: media ± desviación entre semillas por configuración.

Lee `experiments/vision/*/results.json` (el nombre de cada corrida termina en `_s<semilla>` salvo la
semilla 42 original) y agrupa por (modelo, estrategia, dataset de entrenamiento). Para cada grupo
reporta, en el test de BRACOL y en el de RoCoLe: F1 macro, recall de roya severa y la detección de roya
(leve o severa) frente a no roya (sana u otra afección), con sensibilidad y especificidad.

Salida: `experiments/vision/summary.csv`.

Uso:
    python -m src.vision.summarize
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.common.paths import EXPERIMENTS_DIR

OUT = EXPERIMENTS_DIR / "vision" / "summary.csv"
SEVERE = 2
RUST = (1, 2)
NOT_RUST = (0, 3)


def rust_detection(confusion: list) -> tuple:
    """Sensibilidad y especificidad de 'tiene roya' (leve o severa) a partir de la matriz de 4 clases."""
    cm = np.asarray(confusion)
    tp = cm[np.ix_(RUST, RUST)].sum()
    fn = cm[np.ix_(RUST, NOT_RUST)].sum()
    tn = cm[np.ix_(NOT_RUST, NOT_RUST)].sum()
    fp = cm[np.ix_(NOT_RUST, RUST)].sum()
    return tp / (tp + fn), tn / (tn + fp)


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted((EXPERIMENTS_DIR / "vision").glob("*/results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        cfg = r["configuracion"]
        if "smoke" in path.parent.name:
            continue
        row = {"modelo": cfg["model"], "estrategia": cfg["strategy"], "entrenado_con": cfg["train_on"],
               "semilla": cfg["seed"], "corrida": r["nombre"], "f1_val": r["f1_macro_val"]}
        for dataset in ("bracol", "rocole"):
            test = r["test"][dataset]
            sens, spec = rust_detection(test["matriz_confusion"])
            row.update({f"{dataset}_f1": test["f1_macro"],
                        f"{dataset}_recall_severa": test["recall_por_clase"]["roya_severa"],
                        f"{dataset}_sens_roya": sens, f"{dataset}_esp_roya": spec})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["modelo", "estrategia", "entrenado_con"]
    metrics = [c for c in runs.columns if c.startswith(("bracol_", "rocole_"))]
    grouped = runs.groupby(keys)
    mean, std = grouped[metrics].mean(), grouped[metrics].std()
    out = pd.concat({"media": mean, "desv": std}, axis=1)
    out.insert(0, "semillas", grouped["semilla"].nunique())
    return out


def main() -> None:
    runs = load_runs()
    table = summarize(runs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT)
    pd.set_option("display.width", 220)
    print("Por corrida:")
    print(runs[["corrida", "bracol_f1", "rocole_f1"]].round(3).to_string(index=False))
    print("\nMedia ± desviación entre semillas (semillas = n):")
    for key, r in table.iterrows():
        def fmt(metric):
            m, s = r[("media", metric)], r[("desv", metric)]
            return f"{m:.3f}" + (f"±{s:.3f}" if not np.isnan(s) else "")
        print(f"  {key[0]:16s} {key[1]:20s} {key[2]:7s} n={int(r['semillas'])} | BRACOL F1 {fmt('bracol_f1')} "
              f"sev {fmt('bracol_recall_severa')} roya sens {fmt('bracol_sens_roya')} esp {fmt('bracol_esp_roya')} "
              f"|| RoCoLe F1 {fmt('rocole_f1')} sev {fmt('rocole_recall_severa')} "
              f"roya sens {fmt('rocole_sens_roya')} esp {fmt('rocole_esp_roya')}")


if __name__ == "__main__":
    main()
