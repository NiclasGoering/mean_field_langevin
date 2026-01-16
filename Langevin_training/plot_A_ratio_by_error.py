from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results_icml2027/SNR_1501_5")
eps = 1e-12
strict = True

pattern = re.compile(
    r"P_(?P<P>\d+)_d_(?P<d>\d+)_k_(?P<k>\d+)_exp_(?P<exp>\d+)_"
    r"kappa_(?P<kappa>[-+eE0-9.]+)_eta0_(?P<eta0>[-+eE0-9.]+)_"
    r"etaf_(?P<etaf>[-+eE0-9.]+)_gamma_(?P<gamma>[-+eE0-9.]+)"
)


def parse_meta(path: Path):
    m = pattern.search(path.stem)
    if not m:
        return None
    meta = m.groupdict()
    out = {}
    for k, v in meta.items():
        if k in {"kappa", "eta0", "etaf", "gamma"}:
            out[k] = float(v.rstrip("."))
        else:
            out[k] = int(v)
    return out


def meta_key(m):
    return (m["P"], m["d"], m["k"], m["exp"], m["kappa"], m["eta0"], m["etaf"], m["gamma"])


def main():
    diag_paths = sorted(base_dir.rglob("diagnostics_*.json"))
    metrics_paths = sorted(base_dir.rglob("metrics_*.csv"))

    print(f"Found {len(diag_paths)} diagnostics files")
    print(f"Found {len(metrics_paths)} metrics files")

    diag_map = {}
    for p in diag_paths:
        meta = parse_meta(p)
        if meta is None:
            continue
        diag_map[p] = meta

    metrics_map = {}
    for p in metrics_paths:
        meta = parse_meta(p)
        if meta is None:
            continue
        metrics_map[p] = meta

    print(f"Parsed {len(diag_map)} diagnostics metas")
    print(f"Parsed {len(metrics_map)} metrics metas")

    metrics_by_key = {meta_key(m): p for p, m in metrics_map.items()}

    run_rows = []
    for diag_path, meta in diag_map.items():
        key = meta_key(meta)
        metrics_path = metrics_by_key.get(key)
        if metrics_path is None:
            if strict:
                continue
            final_test_err = np.nan
        else:
            dfm = pd.read_csv(metrics_path)
            final_test_err = float(dfm["test_error_01"].iloc[-1]) if len(dfm) else np.nan

        with open(diag_path, "r") as f:
            diag = json.load(f)
        diag_log = diag.get("diag_log", [])
        if not diag_log:
            continue

        df = pd.DataFrame(diag_log)
        if "A_on_abs" not in df or "A_off_abs" not in df:
            continue
        df["A_ratio"] = df["A_on_abs"] / (df["A_off_abs"] + eps)
        if "snr_on" in df and "snr_off" in df:
            df["snr_ratio"] = df["snr_on"] / (df["snr_off"] + eps)
        df["final_test_error_01"] = final_test_err

        for k, v in meta.items():
            df[k] = v

        df["run_id"] = (
            f"P{meta['P']}_d{meta['d']}_k{meta['k']}_exp{meta['exp']}"
            f"_kappa{meta['kappa']}_eta0{meta['eta0']}_etaf{meta['etaf']}_gamma{meta['gamma']}"
        )
        run_rows.append(df)

    if not run_rows:
        raise RuntimeError("No runs found. Check base_dir or filename pattern.")

    runs_df = pd.concat(run_rows, ignore_index=True)
    if "test_error_01" in runs_df:
        runs_df["test_error_drop"] = runs_df.groupby("run_id")["test_error_01"].transform(lambda s: s.iloc[0] - s)

    groups = list(runs_df.groupby("run_id"))
    final_errs = [g[1]["final_test_error_01"].iloc[-1] for g in groups]
    vmin, vmax = np.nanmin(final_errs), np.nanmax(final_errs)
    cmap = plt.cm.viridis

    panels = [
        ("train_mse", "train MSE"),
        ("test_mse", "test MSE"),
        ("train_error_01", "train 0-1 error"),
        ("test_error_01", "test 0-1 error"),
        ("snr_and_test_drop", "SNR on/off + test error drop"),
        ("snr_on", "SNR on"),
        ("snr_off", "SNR off"),
        ("snr_ratio", "SNR on/off ratio"),
        ("max_snr_on_q50", "max SNR on q50"),
        ("max_snr_on_q90", "max SNR on q90"),
        ("A_on_abs", "A on abs"),
        ("A_off_abs", "A off abs"),
        ("A_tilde_on_abs", "A~ on abs"),
        ("A_tilde_off_abs", "A~ off abs"),
        ("A_tilde_abs_on_abs", "A~ |a| on abs"),
        ("A_tilde_abs_off_abs", "A~ |a| off abs"),
        ("A_ratio", "A on/off ratio"),
        ("A_tilde_ratio", "A~ on/off ratio"),
        ("A_tilde_abs_ratio", "A~ |a| on/off ratio"),
        ("anisotropy_ratio", "anisotropy ratio"),
        ("winner_fraction_gt1", "winner fraction > 1"),
        ("c_mode", "c_mode"),
        ("snr_mode_mean", "snr_mode_mean"),
        ("snr_mode_q90", "snr_mode_q90"),
        ("residual_coupling", "residual coupling"),
        ("gate_polarized_fraction", "gate polarized fraction"),
        ("gate_entropy", "gate entropy"),
        ("corr_snr_ratio_test_drop", "corr(SNR ratio, test err drop)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.0 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    ax2_map = {}

    for (gkey, gdf), ferr in zip(groups, final_errs):
        gdf = gdf.sort_values("epoch")
        color = cmap((ferr - vmin) / (vmax - vmin + 1e-12))
        for ax, (col, label) in zip(axes, panels):
            if col == "snr_and_test_drop":
                if "snr_on" not in gdf or "snr_off" not in gdf or "test_error_01" not in gdf:
                    continue
                test_err_drop = gdf["test_error_01"].iloc[0] - gdf["test_error_01"]
                ax.plot(gdf["epoch"], gdf["snr_on"], color=color, alpha=0.35)
                ax.plot(gdf["epoch"], gdf["snr_off"], color=color, alpha=0.35, linestyle="--")
                ax.set_ylabel(label)
                ax2 = ax2_map.get(ax)
                if ax2 is None:
                    ax2 = ax.twinx()
                    ax2.set_ylabel("test err drop")
                    ax2_map[ax] = ax2
                ax2.plot(gdf["epoch"], test_err_drop, color=color, alpha=0.6, linewidth=0.8)
            elif col == "corr_snr_ratio_test_drop":
                continue
            else:
                if col not in gdf:
                    continue
                ax.plot(gdf["epoch"], gdf[col], color=color, alpha=0.4)
                ax.set_ylabel(label)

    axes[-1].set_xlabel("epoch")
    axes[-1].set_xscale("log")

    corr_ax = axes[-1]
    corr_ax.set_ylabel("corr(SNR ratio, test err drop)")
    if "snr_ratio" in runs_df and "test_error_drop" in runs_df:
        corr_rows = []
        for epoch, edf in runs_df.groupby("epoch"):
            x = edf["snr_ratio"].to_numpy()
            y = edf["test_error_drop"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                corr = np.nan
            else:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
            corr_rows.append((epoch, corr))
        corr_rows.sort(key=lambda t: t[0])
        corr_epochs = [r[0] for r in corr_rows]
        corr_vals = [r[1] for r in corr_rows]
        corr_ax.plot(corr_epochs, corr_vals, color="black", linewidth=1.0)

    fig.subplots_adjust(right=0.88, hspace=0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax, label="final test error 0-1")
    out_path = base_dir / "all_timeseries_by_error.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
