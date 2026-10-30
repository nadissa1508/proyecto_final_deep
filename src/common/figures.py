"""Regenera las figuras del informe en `docs/figures/` a partir de `experiments/` (no se escribe ninguna cifra a mano).

    python -m src.common.figures                 # todas
    python -m src.common.figures --only voi loso # algunas

Cada función lee un archivo de resultados y falla con un mensaje claro si no existe. Paleta categórica validada de dataviz
(azul, naranja, aguamarina; solo se usan tres series por gráfica) y etiquetas directas o leyenda en cada figura.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

EXP = Path("experiments")
OUT = Path("docs") / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
VAR_ES = {"temperature_2m_mean": "Temperatura (°C)", "relative_humidity_2m_mean": "Humedad (puntos)", "precipitation_sum": "Lluvia (mm/día)"}
VAR_SHORT = {"temperature_2m_mean": "temperatura", "relative_humidity_2m_mean": "humedad", "precipitation_sum": "lluvia"}
CLASES = ["sana", "roya leve", "roya severa", "otra afección"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "figure.dpi": 100, "font.family": "DejaVu Sans",
    })


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}: corra antes el paso que lo genera (ver docs/reproducir.md).")
    return path


def _save(fig, name: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ───────────────────────── visión ─────────────────────────
def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Cada fila suma 1 (recall por clase); filas vacías quedan en 0."""
    m = np.asarray(matrix, dtype=float)
    sums = m.sum(axis=1, keepdims=True)
    return np.divide(m, sums, out=np.zeros_like(m), where=sums > 0)


def fig_confusion(exp: Path = EXP, out: Path = OUT) -> Path:
    data = json.loads(_need(exp / "vision" / "confusion_hoja.json").read_text(encoding="utf-8"))
    cmap = LinearSegmentedColormap.from_list("azul", [SURFACE, BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, key, title in zip(axes, ("test_bracol", "test_rocole"), ("BRACOL (arábica)", "RoCoLe (campo)")):
        counts = np.asarray(data[key])
        norm = row_normalize(counts)
        ax.grid(False)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                ax.text(j, i, f"{int(counts[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color=SURFACE if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), CLASES, rotation=20, ha="right")
        ax.set_yticks(range(4), CLASES)
        ax.set_xlabel("clase predicha")
        ax.set_ylabel("clase real")
        ax.set_title(f"{title} · n = {int(counts.sum())}")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Matriz de confusión por hoja del modelo elegido ({data.get('candidato', '')}); cada celda: hojas y % de la fila", fontsize=10.5, color=INK2)
    return _save(fig, "fig_confusion_vision.png", out)


def fig_vision_candidates(exp: Path = EXP, out: Path = OUT) -> Path:
    df = pd.read_csv(_need(exp / "vision" / "final_comparison.csv"))
    labels = [c.split("_", 1)[0] + ". " + c.split("_", 1)[1].replace("_", " ") for c in df["candidato"]]
    cols = [("val_f1", "validación", AQUA), ("bracol_f1", "test BRACOL", BLUE), ("rocole_f1", "test RoCoLe", ORANGE)]
    y = np.arange(len(df))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    for k, (col, name, color) in enumerate(cols):
        vals = df[col].to_numpy()
        ax.barh(y + (k - 1) * h, vals, height=h - 0.03, color=color, label=name)
        for yi, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 macro")
    ax.set_title("Candidatos de visión: F1 macro en validación y en los dos tests")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.2))
    return _save(fig, "fig_vision_candidatos.png", out)


def _curves(ax, files: List[Path], color: str, label: str) -> None:
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        ax.plot(d["epoca"], d["loss_val"], color=color, lw=1.6, alpha=0.9, label=label if i == 0 else None)
        ax.plot(d["epoca"], d["loss_train"], color=color, lw=1.1, ls="--", alpha=0.7)


def fig_loss_curves(exp: Path = EXP, out: Path = OUT) -> Path:
    vis_r = sorted((exp / "vision").glob("*_resnet18_dlr_both*/metrics.csv"))
    vis_e = sorted((exp / "vision").glob("*_efficientnet_b0_dlr_both*/metrics.csv"))
    fc_g = sorted([*(exp / "forecast").glob("*_forecast_gru/metrics.csv"), *(exp / "forecast").glob("*_forecast_gru_s[0-9]/metrics.csv")])
    if not (vis_r and vis_e and fc_g):
        raise FileNotFoundError("Faltan metrics.csv de las corridas de visión o de pronóstico en experiments/.")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    _curves(axes[0], vis_r, BLUE, "ResNet-18")
    _curves(axes[0], vis_e, ORANGE, "EfficientNet-B0")
    axes[0].set_title("Visión: pérdida por época (4 semillas)")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("pérdida")
    axes[0].legend()
    _curves(axes[1], fc_g, AQUA, "GRU")
    axes[1].set_title("Pronóstico (GRU): pérdida por época (4 semillas)")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("pérdida")
    axes[1].legend()
    fig.text(0.5, -0.03, "Línea continua = validación; discontinua = entrenamiento. En el pronóstico la validación sube desde la época 2 (sobreajuste).",
             ha="center", fontsize=8.5, color=INK2)
    return _save(fig, "fig_curvas_perdida.png", out)


# ───────────────────────── pronóstico ─────────────────────────
def fig_forecast_gain(exp: Path = EXP, out: Path = OUT) -> Path:
    df = pd.read_csv(_need(exp / "forecast" / "summary.csv"))
    colors = {"gru": BLUE, "lstm": ORANGE, "transformer": AQUA}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True, sharey=True)
    for ax, var in zip(axes, VAR_ES):
        sub = df[df["variable"] == var]
        for model, color in colors.items():
            s = sub[sub["modelo"] == model].sort_values("horizonte")
            ax.plot(s["horizonte"], s["mejora_pct"], marker="o", color=color, lw=1.8, label=model.upper() if model != "transformer" else "Transformer")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(sorted(sub["horizonte"].unique()))
        ax.set_title(VAR_ES[var])
        ax.set_xlabel("horizonte (días)")
    axes[0].set_ylabel("mejora frente al mejor baseline (%)")
    axes[0].legend()
    fig.suptitle("Pronóstico contra ERA5: mejora porcentual del MAE frente al mejor de cuatro baselines (negativo = pierde)", fontsize=10.5, color=INK2)
    return _save(fig, "fig_pronostico_mejora.png", out)


def _family(series: pd.Series, kind: str, family: str = "forecast_gru") -> pd.Series:
    pat = re.compile(rf"^{re.escape(kind)}\|{family}(_s\d)?$")
    return series.apply(lambda s: bool(pat.match(s)))


def loso_change(df: pd.DataFrame, family: str = "forecast_gru") -> pd.DataFrame:
    """Cambio porcentual del MAE al afinar con otras estaciones frente al preentrenado (media de semillas y regiones)."""
    rows = []
    for var in VAR_ES:
        for h in sorted(df["horizonte"].unique()):
            base = df[(df["variable"] == var) & (df["horizonte"] == h)]
            if base.empty:
                continue
            pre = base[_family(base["predictor"], "preentrenado", family)].groupby("region")["mae"].mean().mean()
            ft = base[_family(base["predictor"], "loso_ft", family)].groupby("region")["mae"].mean().mean()
            rows.append({"variable": var, "horizonte": int(h), "preentrenado": pre, "afinado": ft, "cambio_pct": (ft / pre - 1) * 100})
    return pd.DataFrame(rows)


def fig_loso(exp: Path = EXP, out: Path = OUT) -> Path:
    df = pd.read_csv(_need(exp / "forecast" / "loso_eval.csv"))
    ch = loso_change(df)
    colors = {"temperature_2m_mean": ORANGE, "relative_humidity_2m_mean": AQUA, "precipitation_sum": BLUE}
    hs = sorted(ch["horizonte"].unique())
    x = np.arange(len(hs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
    for k, var in enumerate(VAR_ES):
        s = ch[ch["variable"] == var].sort_values("horizonte")
        bars = ax.bar(x + (k - 1) * w, s["cambio_pct"], width=w - 0.03, color=colors[var], label=VAR_SHORT[var].capitalize())
        for b, v in zip(bars, s["cambio_pct"]):
            ax.text(b.get_x() + b.get_width() / 2, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [f"{h} día" + ("s" if h > 1 else "") for h in hs])
    ax.set_ylabel("cambio del MAE al afinar (%)")
    ax.set_title("Finca nueva: afinar con otras 5 estaciones frente al modelo preentrenado (GRU; negativo = mejora)")
    ax.legend(loc="upper right", ncol=3)
    return _save(fig, "fig_finca_nueva.png", out)


def fig_latency(exp: Path = EXP, out: Path = OUT) -> Path:
    df = pd.read_csv(_need(exp / "forecast" / "latency_staleness.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True, sharey=True)
    palette = {1: ORANGE, 3: AQUA, 7: BLUE}
    for ax, var in zip(axes, VAR_ES):
        sub = df[df["variable"] == var]
        piv = sub.groupby(["k_dias", "horizonte"])["mae"].mean().unstack()
        for h, color in palette.items():
            if h not in piv.columns:
                continue
            pct = (piv[h] / piv.loc[0, h] - 1) * 100
            ax.plot(pct.index, pct.values, marker="o", color=color, lw=1.8, label=f"{h} día" + ("s" if h > 1 else ""))
        ax.set_title(VAR_ES[var])
        ax.set_xlabel("días de retraso de ERA5 (k)")
        ax.set_xticks(sorted(sub["k_dias"].unique()))
    axes[0].set_ylabel("aumento del MAE frente a k = 0 (%)")
    axes[0].legend(title="horizonte", title_fontsize=8.5)
    fig.suptitle("Costo del retraso de ERA5 rellenando por persistencia (GRU, media de regiones y semillas)", fontsize=10.5, color=INK2)
    return _save(fig, "fig_retraso_era5.png", out)


# ───────────────────────── integración ─────────────────────────
VOI_ORDER = [
    ("A_techo", "A. Techo: roya conocida"),
    ("D0_solo_CNN", "D0. Solo fotos (10 hojas)"),
    ("D_pronostico_y_CNN", "D. Pronóstico + fotos"),
    ("C_pronostico", "C. Pronóstico"),
    ("C0_pronostico_perfecto", "C0. Pronóstico perfecto (control)"),
    ("B_clima_observado", "B. Clima observado"),
]


def fig_voi(exp: Path = EXP, out: Path = OUT, retraso: int = 0) -> Path:
    df = pd.read_csv(_need(exp / "fusion" / "voi_resumen.csv"))
    df = df[(df["retraso"] == retraso) & (df["gamma"] == 0.15) & (df["hojas"] == 10) & (df["confusion_mundo"] == "pooled")]
    rows = [(label, df[df["escenario"] == key].iloc[0]) for key, label in VOI_ORDER if (df["escenario"] == key).any()]
    if not rows:
        raise ValueError("voi_resumen.csv no tiene los escenarios esperados.")
    labels = [r[0] for r in rows]
    vals = np.array([r[1]["vs_sin_informacion"] for r in rows])
    errs = np.array([r[1]["error_estandar"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, vals, xerr=errs, color=BLUE, height=0.55, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + (e + 3 if v >= 0 else -e - 3), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(min(vals.min() - 25, -30), vals.max() + 30)
    ax.set_xlabel("ingreso anual frente a no tener información (USD/ha); barras de error = error estándar entre 24 región-años")
    ax.set_title(f"Valor de la información en el mundo simulado (retraso de ERA5: {retraso} días; supuestos declarados, no medición de campo)")
    return _save(fig, "fig_valor_informacion.png", out)


def fig_voi_leaves(exp: Path = EXP, out: Path = OUT) -> Path:
    """Valor de solo fotos según el número de hojas (matriz de confusión agrupada)."""
    df = pd.read_csv(_need(exp / "fusion" / "voi_sensibilidad.csv"))
    sub = df[(df["escenario"] == "D0_solo_CNN") & (df["confusion_mundo"] == "pooled") & (df["retraso"] == 0)]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for (gamma, g), color in zip(sub.groupby("gamma"), (BLUE, ORANGE, AQUA)):
        g = g.sort_values("hojas")
        ax.plot(g["hojas"], g["vs_sin_informacion"], marker="o", ms=5, lw=1.8, color=color, label=f"acople del clima γ = {gamma:g}")
    ax.set_xlabel("hojas fotografiadas")
    ax.set_ylabel("valor frente a no tener información (USD/ha)")
    ax.set_title("Solo fotos: el valor crece con el número de hojas (mundo simulado)")
    ax.legend()
    return _save(fig, "fig_valor_fotos_hojas.png", out)


FIGURES: Dict[str, Callable[..., Path]] = {
    "confusion": fig_confusion, "candidatos": fig_vision_candidates, "curvas": fig_loss_curves,
    "mejora": fig_forecast_gain, "loso": fig_loso, "retraso": fig_latency, "voi": fig_voi, "hojas": fig_voi_leaves,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenera las figuras del informe en docs/figures/.")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="solo estas figuras")
    parser.add_argument("--exp", type=Path, default=EXP)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    _style()
    failed = 0
    for name in args.only or FIGURES:
        try:
            print("OK ", FIGURES[name](args.exp, args.out))
        except Exception as exc:  # una figura que falla no debe ocultar las demás
            failed += 1
            print(f"ERROR {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
