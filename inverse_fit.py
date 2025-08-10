#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inverse fit for droplet deformation from contours/summary CSV.
# Usage:
#   python inverse_fit.py --csv droplet_frames_summary.csv --json new_expirement_3drop_rotated.json --fps 25 --out inverse_out

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    c = np.convolve(x, np.ones(w)/w, mode="same")
    c[:w//2] = np.mean(x[:w])
    c[-w//2:] = np.mean(x[-w:])
    return c


def pca_axes(points: np.ndarray) -> Tuple[float, float, float]:
    if points.shape[0] < 3:
        return np.nan, np.nan, 0.0
    mu = points.mean(axis=0)
    X = points - mu
    C = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    std1 = math.sqrt(max(eigvals[0], 0.0))
    std2 = math.sqrt(max(eigvals[1], 0.0))
    k = 2.0
    a = k * std1
    b = k * std2
    angle = math.atan2(eigvecs[1, 0], eigvecs[0, 0])
    return float(a), float(b), float(angle)


def polygon_area_centroid(contour: np.ndarray) -> Tuple[float, float, float]:
    x = contour[:, 0]
    y = contour[:, 1]
    if not (x[0] == x[-1] and y[0] == y[-1]):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    a = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if abs(a) < 1e-12:
        return 0.0, float(np.mean(x[:-1])), float(np.mean(y[:-1]))
    cx = (1.0 / (6.0 * a)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    cy = (1.0 / (6.0 * a)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(abs(a)), float(cx), float(cy)


def load_summary_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    if "drop_id" in df.columns:
        rename_map["drop_id"] = "series"
    elif "drop" in df.columns:
        rename_map["drop"] = "series"
    elif "id" in df.columns:
        rename_map["id"] = "series"
    if "frame_index" in df.columns:
        rename_map["frame_index"] = "frame"
    if "time_s" in df.columns:
        rename_map["time_s"] = "time"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "series" not in df.columns:
        df["series"] = 0
    if "frame" not in df.columns:
        if "frame_id" in df.columns:
            df = df.rename(columns={"frame_id": "frame"})
        else:
            df["frame"] = np.arange(len(df), dtype=int)
    return df


def parse_json_flex(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = None
    if isinstance(data, dict):
        for key in ["frames", "data", "items", "results"]:
            if key in data and isinstance(data[key], list):
                frames = data[key]
                break
        if frames is None and isinstance(data.get("frame"), (int, float)):
            frames = [data]
    elif isinstance(data, list):
        frames = data
    else:
        frames = []

    records = []
    if not frames:
        return records

    for fobj in frames:
        if isinstance(fobj, dict):
            frame_idx = fobj.get("frame")
            if frame_idx is None:
                for k in ["frame_id", "index", "i"]:
                    if k in fobj:
                        frame_idx = fobj[k]
                        break
            drops = fobj.get("drops") or fobj.get("objects") or fobj.get("detections")
            if drops is None and "contour" in fobj:
                drops = [fobj]
        else:
            frame_idx = None
            drops = None

        if drops is None:
            continue

        for j, dobj in enumerate(drops):
            series = dobj.get("id") or dobj.get("index") or dobj.get("label") or j
            contour = dobj.get("contour") or dobj.get("polygon") or dobj.get("points")
            if contour is None:
                for k in ["segmentation", "mask", "coords"]:
                    if k in dobj:
                        contour = dobj[k]
                        break
            if contour is None:
                continue
            if isinstance(contour, list) and contour and isinstance(contour[0], (int, float)):
                arr = np.asarray(contour, dtype=float).reshape(-1, 2)
            else:
                arr = np.asarray(contour, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                continue
            records.append({"frame": frame_idx if frame_idx is not None else -1,
                            "series": series,
                            "contour": arr})
    return records


def df_from_contours(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        area, cx, cy = polygon_area_centroid(rec["contour"])
        a, b, ang = pca_axes(rec["contour"])
        if not (np.isfinite(a) and np.isfinite(b) and area > 0):
            continue
        rows.append({
            "frame": rec["frame"],
            "series": rec["series"],
            "area": area,
            "a": max(a, b),
            "b": min(a, b),
            "angle": ang,
            "cx": cx,
            "cy": cy
        })
    if not rows:
        return pd.DataFrame(columns=["frame", "series", "area", "a", "b", "angle", "cx", "cy"])
    df = pd.DataFrame(rows)
    try:
        df["series"] = pd.factorize(df["series"])[0]
    except Exception:
        pass
    return df


# ---- NEW: coerce axes/elongation from aliases ----
def coerce_axes_and_elongation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = set(out.columns)

    # Map a/b from common aliases if missing
    major_aliases = [
        "a", "major", "major_axis", "semi_major", "semi_major_axis",
        "majoraxis", "maj_axis", "axis_major", "long_axis", "ellipse_major_axis"
    ]
    minor_aliases = [
        "b", "minor", "minor_axis", "semi_minor", "semi_minor_axis",
        "minoraxis", "min_axis", "axis_minor", "short_axis", "ellipse_minor_axis"
    ]

    def pick_alias(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    if "a" not in cols:
        maj = pick_alias(major_aliases)
        if maj:
            out["a"] = out[maj]
            cols.add("a")
    if "b" not in cols:
        mnr = pick_alias(minor_aliases)
        if mnr:
            out["b"] = out[mnr]
            cols.add("b")

    # Create elongation 'e' if absent
    if "e" not in cols:
        if "a" in out.columns and "b" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                out["e"] = np.where(out["b"] > 0.0, out["a"] / out["b"], np.nan)
        else:
            for cand in ["elongation", "aspect", "aspect_ratio", "ratio", "e_long"]:
                if cand in out.columns:
                    out = out.rename(columns={cand: "e"})
                    break
    return out


def build_features(df: pd.DataFrame,
                   fps: float = 25.0,
                   smooth: bool = True,
                   smooth_win: int = 9,
                   smooth_poly: int = 2) -> pd.DataFrame:
    out = df.copy()

    # Coerce to numeric if present
    if "e" in out.columns:
        out["e"] = pd.to_numeric(out["e"], errors="coerce")
    if "r" in out.columns:
        out["r"] = pd.to_numeric(out["r"], errors="coerce")

    if "time" not in out.columns:
        if "frame" in out.columns:
            out["time"] = out["frame"] / float(fps)
        else:
            out["time"] = np.arange(len(out)) / float(fps)

    if "area" in out.columns and "r" not in out.columns:
        out["r"] = np.sqrt(np.maximum(out["area"], 0.0) / math.pi)
    if "a" in out.columns and "b" in out.columns and "e" not in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["e"] = np.where(out["b"] > 0.0, out["a"] / out["b"], np.nan)

    for cand in ["elongation", "e_long", "aspect", "ratio", "aspect_ratio"]:
        if cand in out.columns and "e" not in out.columns:
            out = out.rename(columns={cand: "e"})
            break

    out = out.sort_values(["series", "time"]).reset_index(drop=True)

    def normalize_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        if "e" not in g.columns:
            g["e"] = np.nan
        if "r" not in g.columns:
            g["r"] = np.nan
        e0 = g["e"].dropna().iloc[0] if g["e"].dropna().size else np.nan
        r0 = g["r"].dropna().iloc[0] if g["r"].dropna().size else np.nan
        g["e_norm"] = g["e"] / e0 if np.isfinite(e0) and e0 != 0 else g["e"]
        g["r_norm"] = g["r"] / r0 if np.isfinite(r0) and r0 != 0 else g["r"]
        if smooth and g["e_norm"].notna().sum() >= max(7, smooth_win):
            arr = g["e_norm"].values
            if savgol_filter is not None:
                win = min(smooth_win if smooth_win % 2 == 1 else smooth_win + 1, len(arr) - (len(arr) + 1) % 2)
                if win < 3:
                    win = 3
                poly = min(smooth_poly, win - 1)
                try:
                    g["e_smooth"] = savgol_filter(arr, window_length=win, polyorder=poly, mode="interp")
                except Exception:
                    g["e_smooth"] = moving_average(arr, w=5)
            else:
                g["e_smooth"] = moving_average(arr, w=5)
        else:
            g["e_smooth"] = g["e_norm"]
        return g

    out = out.groupby("series", group_keys=False).apply(normalize_group)

    if "e" not in out.columns or out["e"].isna().all():
        print("[WARN] Column 'e' is missing or all NaN after processing. "
              "Proceeding anyway; downstream dropna may remove empty rows. "
              f"Columns present: {list(out.columns)[:30]}")
    return out


@dataclass
class FitParams:
    c1: float
    c2: float
    k_bo: float
    tau: float


def model_e_inf(R: np.ndarray, p: FitParams) -> np.ndarray:
    Bo = p.k_bo * R
    return 1.0 + p.c1 * Bo + p.c2 * Bo * Bo


def model_e_t(t: np.ndarray, R: np.ndarray, e0: float, t0: float, p: FitParams) -> np.ndarray:
    e_inf = model_e_inf(R, p)
    return e_inf + (e0 - e_inf) * np.exp(-(t - t0) / max(p.tau, 1e-6))


def pack_params(x: np.ndarray) -> FitParams:
    return FitParams(c1=float(x[0]), c2=float(x[1]), k_bo=float(x[2]), tau=float(x[3]))


def residuals_global(x: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    p = pack_params(x)
    resids = []
    for s, g in df.groupby("series"):
        g = g.dropna(subset=["time", "r_norm", "e_smooth"])
        if g.empty:
            continue
        t = g["time"].values
        R = g["r_norm"].values
        e = g["e_smooth"].values
        e0 = e[0]
        t0 = t[0]
        pred = model_e_t(t, R, e0=e0, t0=t0, p=p)
        resids.append((e - pred))
    if not resids:
        return np.array([], dtype=float)
    return np.concatenate(resids, axis=0)


def fit_parameters(df: pd.DataFrame,
                   x0: Optional[np.ndarray] = None,
                   bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[FitParams, Dict]:
    if x0 is None:
        x0 = np.array([0.8, 0.1, 0.5, 0.2], dtype=float)
    if bounds is None:
        lb = np.array([0.0, -1.0, 0.0, 1e-4])
        ub = np.array([3.0,  1.0, 10.0, 60.0])
        bounds = (lb, ub)

    if least_squares is None:
        print("[WARN] scipy.optimize.least_squares not available. Using basic random search.")
        best_x = x0.copy()
        best_loss = np.inf
        rng = np.random.default_rng(42)
        for it in range(3000):
            cand = best_x + rng.normal(scale=[0.1, 0.05, 0.1, 0.05], size=4)
            cand = np.minimum(np.maximum(cand, bounds[0]), bounds[1])
            r = residuals_global(cand, df)
            loss = float(np.sum(r*r))
            if loss < best_loss:
                best_loss = loss
                best_x = cand
        params = pack_params(best_x)
        info = {"method": "random_search", "loss": best_loss}
        return params, info

    res = least_squares(residuals_global, x0=x0, bounds=bounds, args=(df,), jac="2-point", verbose=1)
    params = pack_params(res.x)
    cov = None
    try:
        JtJ = res.jac.T @ res.jac
        cov = np.linalg.pinv(JtJ)
    except Exception:
        pass
    info = {
        "cost": float(res.cost),
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "njev": int(getattr(res, "njev", 0)),
        "covariance": cov.tolist() if cov is not None else None
    }
    return params, info


def plot_series_fit(g: pd.DataFrame, p: FitParams, out_path: str):
    t = g["time"].values
    R = g["r_norm"].values
    e = g["e_norm"].values
    e_s = g["e_smooth"].values
    e0 = e_s[0]
    t0 = t[0]
    pred = model_e_t(t, R, e0=e0, t0=t0, p=p)

    plt.figure(figsize=(7, 4))
    plt.title(f"Series {int(g['series'].iloc[0])} – E(t)")
    plt.plot(t, e, ".", label="E (raw)")
    plt.plot(t, e_s, "-", label="E (smooth)")
    plt.plot(t, pred, "--", label="model")
    plt.xlabel("time, s")
    plt.ylabel("elongation (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_parameters(p: FitParams, info: Dict, out_dir: str):
    d = {"c1": p.c1, "c2": p.c2, "k_bo": p.k_bo, "tau": p.tau, "fit_info": info}
    with open(os.path.join(out_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def save_predictions(df: pd.DataFrame, p: FitParams, out_dir: str):
    rows = []
    for s, g in df.groupby("series"):
        t = g["time"].values
        R = g["r_norm"].values
        e_s = g["e_smooth"].values
        e0 = e_s[0]
        t0 = t[0]
        pred = model_e_t(t, R, e0=e0, t0=t0, p=p)
        tmp = g.copy()
        tmp["e_pred"] = pred
        rows.append(tmp)
    res = pd.concat(rows, axis=0).reset_index(drop=True)
    res.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="droplet_frames_summary.csv", help="Path to summary CSV (per-frame/per-drop).")
    ap.add_argument("--json", type=str, default="new_expirement_3drop_rotated.json", help="Path to contours JSON (used to compute metrics if missing in CSV).")
    ap.add_argument("--fps", type=float, default=25.0, help="Frames per second if time column missing.")
    ap.add_argument("--out", type=str, default="inverse_out", help="Output directory.")
    ap.add_argument("--no_smooth", action="store_true", help="Disable smoothing of E(t).")
    args = ap.parse_args()

    ensure_dir(args.out)

    if os.path.isfile(args.csv):
        df_csv = load_summary_csv(args.csv)
        print(f"[INFO] Loaded CSV: {args.csv} -> {len(df_csv)} rows")
    else:
        print(f"[WARN] CSV not found: {args.csv}")
        df_csv = pd.DataFrame()

    df_json = pd.DataFrame()
    if os.path.isfile(args.json):
        recs = parse_json_flex(args.json)
        if recs:
            df_json = df_from_contours(recs)
            print(f"[INFO] Parsed JSON contours: {args.json} -> {len(df_json)} rows (series x frames)")
        else:
            print(f"[WARN] Could not parse contours from JSON: {args.json}")
    else:
        print(f"[WARN] JSON not found: {args.json}")

    if not df_csv.empty and not df_json.empty:
        merged = pd.merge(df_csv, df_json, on=["series", "frame"], how="outer", suffixes=("", "_json"))
        for col in ["area", "a", "b", "angle", "cx", "cy"]:
            merged[col] = merged[col].fillna(merged.get(f"{col}_json"))
        df = merged
    elif not df_csv.empty:
        df = df_csv
    else:
        df = df_json

    if df.empty:
        raise SystemExit("No data available from CSV or JSON. Aborting.")

    # NEW: coerce aliases and try to build 'e'
    df = coerce_axes_and_elongation(df)
    print(f"[DEBUG] Columns after merge/coercion: {list(df.columns)[:30]}")

    df_feat = build_features(df, fps=args.fps, smooth=not args.no_smooth)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "r_norm", "e_norm", "e_smooth"])

    params, info = fit_parameters(df_feat)

    os.makedirs(args.out, exist_ok=True)
    save_parameters(params, info, args.out)
    for s, g in df_feat.groupby("series"):
        plot_series_fit(g, params, os.path.join(args.out, f"series_{int(s)}.png"))
    save_predictions(df_feat, params, args.out)

    with open(os.path.join(args.out, "fit_report.txt"), "w", encoding="utf-8") as f:
        f.write("Inverse fit parameters (normalized model)\n")
        f.write(f"c1   = {params.c1:.6g}\n")
        f.write(f"c2   = {params.c2:.6g}\n")
        f.write(f"k_bo = {params.k_bo:.6g}\n")
        f.write(f"tau  = {params.tau:.6g} s\n\n")
        f.write("Notes:\n")
        f.write("- E_inf = 1 + c1*(k_bo*R) + c2*(k_bo*R)^2, with R normalized to first frame per series.\n")
        f.write("- E(t)  = E_inf + (E0 - E_inf) * exp(-(t - t0)/tau), E0 = E_smooth at first frame.\n")
        f.write("- To interpret physically, relate k_bo to (mu0*chi*H^2/gamma).\n")
        f.write("- If residuals are structured, consider tau depending on R or a bi-exponential model.\n")
    print(f"[DONE] Saved outputs to: {args.out}")
    print("  - parameters.json")
    print("  - fit_report.txt")
    print("  - predictions.csv")
    print("  - series_*.png")


if __name__ == "__main__":
    main()
