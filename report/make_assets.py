"""Build the numbers, tables and figures that report.tex includes, from the files in results/.

    python report/make_assets.py          (run from the project root, after the steps in commands.md)

Writes report/generated/*.tex and report/figures/*.png. Every number in the report comes from
a results file written by this project's code; nothing is typed in by hand. Missing results
(e.g. no captured photos yet) are shown as "pending" instead of failing.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.metrics import ces  # noqa: E402

RESULTS, GEN, FIG = ROOT / "results", ROOT / "report" / "generated", ROOT / "report" / "figures"
REFERENCE_TRAIN_LOG = ROOT / "sample-code" / "results" / "train_final.json"


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def f4(x):
    return f"{x:.4f}"


def numbers(metrics, timing, ref, tuning, profile):
    out = ""
    out += macro("CPUName", tex_escape(timing["cpu"]) if timing else "pending")
    out += macro("OSName", tex_escape(timing["os"]) if timing else "pending")
    out += macro("TorchVersion", tex_escape(timing["torch"]) if timing else "pending")
    out += macro("TcpuMs", f"{timing['T_CPU_ms']:.2f}" if timing else "pending")
    out += macro("TcpuN", f"{timing['n']:,}" if timing else "pending")
    s_disk = (metrics or {}).get("S_disk_bytes") or (ref or {}).get("S_disk_bytes")
    out += macro("SdiskBytes", f"{s_disk:,}" if s_disk else "pending")
    out += macro("SdiskMiB", f"{s_disk / 2**20:.3f}" if s_disk else "pending")
    out += macro("NParams", f"{ref['params']:,}" if ref else "pending")
    out += macro("CES", f4(metrics["CES"]) if metrics and "CES" in metrics else "pending")
    for key, name in (("val", "Val"), ("te", "Te"), ("captured", "Cap")):
        s = (metrics or {}).get(key)
        for m in ("mIoU", "BF1", "Q"):
            out += macro(f"{name}{m.replace('mIoU', 'Miou').replace('BF1', 'BFone')}", f4(s[m]) if s else "pending")
        out += macro(f"{name}N", f"{s['n']:,}" if s else "pending")
    if metrics and "CES" in metrics and timing:
        q = metrics["te"]["Q"]
        out += macro("TimePenalty", f"{0.20 * timing['T_CPU_ms'] / 50:.4f}")
        out += macro("SizePenalty", f"{0.10 * min(1, s_disk / 4194304):.4f}")
        out += macro("CESCheck", f"{ces(q, timing['T_CPU_ms'], s_disk):.4f}")
    if tuning:
        d, b = tuning["default"], tuning["best"]
        out += macro("TuneDefaultQ", f4(d["Q"])) + macro("TuneDefaultCES", f4(d["val_CES"]))
        out += macro("TuneBestQ", f4(b["Q"])) + macro("TuneBestCES", f4(b["val_CES"]))
        out += macro("TuneBestThr", f"{b['threshold']:+.1f}") + macro("TuneBestTTA", "on" if b["flip_tta"] else "off")
        out += macro("TuneN", f"{tuning['n_val']:,}")
    if profile:
        out += macro("LayoutContig", f"{profile['layout_ms']['contiguous']:.2f}")
        out += macro("LayoutCL", f"{profile['layout_ms']['channels_last']:.2f}")
    return out


def table_results(metrics):
    rows = []
    for key, label in (("val", "Validation (held out from DUTS-TR)"), ("te", "DUTS-TE"), ("captured", "Own photos")):
        s = (metrics or {}).get(key)
        rows.append(f"{label} & {s['n']:,} & {f4(s['mIoU'])} & {f4(s['BF1'])} & {f4(s['Q'])} \\\\" if s else
                    f"{label} & -- & \\multicolumn{{3}}{{c}}{{pending}} \\\\")
    return "\n".join(rows) + "\n"


def table_bench():
    path = RESULTS / "bench.md"
    if not path.exists():
        return "\\multicolumn{6}{c}{pending: run python -m src.bench} \\\\\n"
    tta = {}
    if (RESULTS / "bench_tta.md").exists():
        for line in (RESULTS / "bench_tta.md").read_text().splitlines():
            c = [x.strip() for x in line.strip("|").split("|")]
            if line.startswith("| r"):
                tta[c[0]] = c[6]
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("| r"):
            continue
        name, widths, depths, res, params, size, ms = [x.strip() for x in line.strip("|").split("|")]
        rows.append(f"{tex_escape(name)} & {widths} / {depths} & {res} & {params} & {size} & {ms} & {tta.get(name, '--')} \\\\")
    return "\n".join(rows) + "\n"


def table_stages(profile):
    if not profile:
        return "\\multicolumn{2}{c}{pending: run python -m src.profile\\_stages} \\\\\n"
    labels = {"downscale": "Downscale $256\\to128$ (\\texttt{cv2.INTER\\_AREA})", "to_tensor": "uint8 array $\\to$ float tensor",
              "network": "Network (TorchScript, frozen)", "upsample_threshold": "Upsample logit $64\\to256$, threshold"}
    st = profile["stage_ms"]
    rows = [f"{labels[k]} & {v:.3f} \\\\" for k, v in st.items()]
    rows.append(f"\\midrule Total & {sum(st.values()):.3f} \\\\")
    return "\n".join(rows) + "\n"


def table_tuning(tuning):
    if not tuning:
        return "\\multicolumn{6}{c}{pending: run python -m src.tune\\_deployed} \\\\\n"
    rows = []
    for tta in (False, True):
        sub = [r for r in tuning["rows"] if r["flip_tta"] == tta]
        for t in sorted({r["threshold"] for r in sub}):
            r = next(x for x in sub if x["threshold"] == t)
            if abs(t) <= 1.0 + 1e-9 and round(t * 10) % 2 == 0:  # every 0.2 within [-1, 1]
                best = r is tuning["best"] or (r["threshold"] == tuning["best"]["threshold"] and tta == tuning["best"]["flip_tta"])
                cells = [f"{'on' if tta else 'off'}", f"{t:+.1f}", f4(r["mIoU"]), f4(r["BF1"]), f4(r["Q"]),
                         f"{r['T_ms']:.2f}", f4(r["val_CES"])]
                if best:
                    cells = [f"\\textbf{{{c}}}" for c in cells]
                rows.append(" & ".join(cells) + " \\\\")
        if not tta:
            rows.append("\\midrule")
    return "\n".join(rows) + "\n"


def table_worst(per_image, k=8):
    if not per_image:
        return "\\multicolumn{3}{c}{pending} \\\\\n"
    worst = sorted(per_image, key=lambda r: (r["IoU_obj"], r["BF1"]))[:k]
    return "\n".join(f"\\texttt{{{tex_escape(r['name'])}}} & {r['IoU_obj']:.3f} & {r['BF1']:.3f} \\\\" for r in worst) + "\n"


def te_stats(per_image):
    if not per_image:
        return ""
    iou = np.array([r["IoU_obj"] for r in per_image])
    out = macro("TeShareIoULow", f"{100 * (iou < 0.1).mean():.1f}")
    out += macro("TeShareIoUHigh", f"{100 * (iou >= 0.9).mean():.1f}")
    out += macro("TeMedianIoU", f"{np.median(iou):.3f}")
    return out


def fig_iou_hist(per_image_te, per_image_val):
    if not per_image_te:
        return
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bins = np.linspace(0, 1, 26)
    for rows, label in ((per_image_val, "validation"), (per_image_te, "DUTS-TE")):
        if rows:
            ax.hist([r["IoU_obj"] for r in rows], bins=bins, density=True, alpha=0.55, label=label)
    ax.set(xlabel="per-photo object IoU", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "iou_hist.png", dpi=160)
    plt.close(fig)


def fig_reference_curve():
    if not REFERENCE_TRAIN_LOG.exists():
        return
    hist = json.loads(REFERENCE_TRAIN_LOG.read_text())["history"]
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.8))
    a.plot(ep, [h["loss"] for h in hist])
    a.set(xlabel="epoch", ylabel="training loss")
    for key in ("mIoU", "BF1", "Q"):
        b.plot(ep, [h[key] for h in hist], label=key)
    b.set(xlabel="epoch", ylabel="validation score")
    b.legend()
    fig.tight_layout()
    fig.savefig(FIG / "curve_reference.png", dpi=160)
    plt.close(fig)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    metrics, timing, ref = load("metrics.json"), load("timing.json"), load("reference_model.json")
    tuning, profile = load("tuning_model.json"), load("profile_stages.json")
    per_te, per_val = load("per_image_te.json"), load("per_image_val.json")

    (GEN / "numbers.tex").write_text(numbers(metrics, timing, ref, tuning, profile) + te_stats(per_te)
                                     + macro("HaveCaptured", "1" if metrics and "captured" in metrics else "0"))
    (GEN / "tab_results.tex").write_text(table_results(metrics))
    (GEN / "tab_bench.tex").write_text(table_bench())
    (GEN / "tab_stages.tex").write_text(table_stages(profile))
    (GEN / "tab_tuning.tex").write_text(table_tuning(tuning))
    (GEN / "tab_worst_te.tex").write_text(table_worst(per_te))
    fig_iou_hist(per_te, per_val)
    fig_reference_curve()
    print(f"wrote {GEN} and {FIG}")


if __name__ == "__main__":
    main()
