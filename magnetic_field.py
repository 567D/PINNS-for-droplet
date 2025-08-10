"""
PINN inverse (γ, Δρg, magnetic scale) from JSON contours with isotropic normalization
and optional magnetic field map (Hx.npy/Hy.npy). If field maps are missing, uses a
constant field H0 along the chosen axis.

- Input JSON format:
{
  "frames": [
    {"t": 0, "contour": [[x0,y0], ...]},
    {"t": 1, "contour": [[...]]
  ],
  "meta": {"width": 1194, "height": 768}
}

- Geometry (must set):
  Lx, Ly: physical domain size in meters (e.g., Lx=0.03, Ly=0.01)
  H, W   : image size in pixels (e.g., H=768, W=1194) — must match JSON meta

- Magnetic field options:
  1) Load Hx.npy, Hy.npy with shapes (T, H, W) or (1, H, W)
  2) If not found, use constant field H0 along AXIS ('x' or 'y')

Outputs:
  - pinn_inverse_from_json.pt   (model state)
  - training log printed
  - optional preview image "pinn_iso_contour.png"
"""

from __future__ import annotations
import json, math, os, sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ===============
# USER SETTINGS
# ===============
JSON_PATH = "new_expirement_3drop_rotated.json"  # <-- set your file
# Image geometry (must match JSON meta)
H_img, W_img = 768, 1194
Lx, Ly = 0.03, 0.01           # meters

# Field maps (optional). If not present, constant field is used.
FIELD_PATH_HX = "Hx.npy"      # (T,H,W) or (1,H,W)
FIELD_PATH_HY = "Hy.npy"
CONST_FIELD_AXIS = 'y'        # 'x' or 'y' if constant field is used
CONST_FIELD_H0   = 1.0        # A/m (scale can be learned by alpha anyway)

# Training
MAX_ITERS = 12000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123

# Loss weights
W_DATA = 6.0
W_EIK  = 1.0
W_PHYS = 3.0
W_AREA = 0.05
SIGMA_IF = 0.02               # width for interface weighting
GRID_N_AREA = 84              # grid for area approx
BATCH_CONTOUR = 4096

rng_np = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# =========================
# Geometry / normalization
# =========================
H, W = H_img, W_img
dx, dy = Lx / W, Ly / H               # meters per pixel (anisotropic)
s_iso = max(Lx, Ly) / 2.0             # isotropic scale for normalization


def polygon_area_meters(cnt_px: np.ndarray, dx: float, dy: float) -> float:
    """Signed area in meters^2, cnt_px shape (N,2) in pixels."""
    x = cnt_px[:, 0] * dx
    y = cnt_px[:, 1] * dy
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def norm_xy_px_iso(cnt_px: np.ndarray) -> np.ndarray:
    """Convert pixel contour to isotropic, dimensionless coords using s_iso.
    Returns array (N,2).
    """
    xs = cnt_px[:, 0] * dx / s_iso
    ys = cnt_px[:, 1] * dy / s_iso
    return np.stack([xs, ys], 1).astype(np.float32)


@dataclass
class FrameData:
    t: float
    tn: float
    contour_iso: np.ndarray  # (Nc,2)
    area_iso: float


def load_json_iso(json_path: str) -> Tuple[List[FrameData], Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        J = json.load(f)

    # размеры кадра из данных карты поля, если есть, иначе из настроек
    global dx, dy, H_img, W_img
    if isinstance(J, dict) and "meta" in J and "width" in J["meta"] and "height" in J["meta"]:
        Wj = int(J["meta"]["width"])
        Hj = int(J["meta"]["height"])
    elif 'field_size' in globals() and isinstance(field_size, (tuple, list)) and len(field_size) == 2:
        Hj, Wj = field_size
        print(f"[INFO] JSON has no 'meta'; using field map size H={Hj}, W={Wj}")
    elif 'Hx' in globals() and isinstance(Hx, torch.Tensor) and Hx.ndim == 4:
        # если есть загруженная карта поля, берём её размеры
        _, _, Hj, Wj = Hx.shape
        print(f"[INFO] JSON has no 'meta'; using Hx.npy size H={Hj}, W={Wj}")
    else:
        Wj, Hj = W_img, H_img
        print(f"[INFO] JSON has no 'meta'; using script settings H={H_img}, W={W_img}")

    dx = Lx / Wj
    dy = Ly / Hj

    def frame_key_to_int(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    frame_keys = sorted([k for k in J.keys() if k.startswith("frame_")], key=frame_key_to_int)
    if not frame_keys and "frames" in J:
        frame_keys = ["frames"]

    frames: List[FrameData] = []
    ts_abs = []

    for fidx, k in enumerate(frame_keys):
        items = J[k] if isinstance(J[k], list) else J["frames"]
        if not items:
            continue

        for didx, it in enumerate(items):
            poly = np.array(it.get("polygon", it.get("contour")), dtype=np.float32)
            if poly.ndim != 2 or poly.shape[1] != 2:
                continue

            if poly.max() <= 1.0:
                poly[:, 0] *= (Wj - 1)
                poly[:, 1] *= (Hj - 1)

            area_m = abs(polygon_area_meters(poly, dx, dy))
            area_iso = area_m / (s_iso ** 2)

            xs_iso = (poly[:, 0] * dx) / s_iso
            ys_iso = (poly[:, 1] * dy) / s_iso
            cnt_iso = np.stack([xs_iso, ys_iso], 1).astype(np.float32)

            t_abs = float(fidx)
            frames.append(FrameData(t=t_abs, tn=0.0, contour_iso=cnt_iso, area_iso=area_iso))
            ts_abs.append(t_abs)

    if frames:
        tmin, tmax = min(ts_abs), max(ts_abs)
        for fr in frames:
            fr.tn = 0.0 if tmax == tmin else (fr.t - tmin) / (tmax - tmin)

    meta = {"H": Hj, "W": Wj, "Lx": Lx, "Ly": Ly, "dx": dx, "dy": dy, "s_iso": s_iso}
    return frames, meta





# =========================
# Field maps (Hx, Hy) loader
# =========================
class FieldProvider:
    def __init__(self, Hx: torch.Tensor | None, Hy: torch.Tensor | None,
                 const_axis: str = 'y', const_H0: float = 1.0, T_expected: int = 1,
                 field_H: int | None = None, field_W: int | None = None):
        """
        Field provider that supports:
        - Constant field (no tensors)
        - Precomputed maps with arbitrary spatial resolution (Hf,Wf),
          possibly different from image size. We assume the same physical Lx,Ly.
        Tensors shape must be (T,1,Hf,Wf) or (1,1,Hf,Wf).
        """
        self.const = Hx is None or Hy is None
        self.axis = const_axis
        self.H0 = const_H0
        self.T = T_expected
        if not self.const:
            self.Hx = Hx  # (T,1,Hf,Wf)
            self.Hy = Hy
            # infer field grid size
            self.field_T, _, self.field_H, self.field_W = self.Hx.shape
        else:
            self.Hx = None
            self.Hy = None
            self.field_H = field_H
            self.field_W = field_W

    @staticmethod
    def from_paths_or_const(path_x: str, path_y: str, T_expected: int, H: int, W: int,
                             device: str, const_axis: str, const_H0: float):
        def _try_load(p):
            if not os.path.exists(p):
                return None
            arr = np.load(p)
            # Accept (T,1,H,W) or (1,H,W) or (H,W)
            if arr.ndim == 4:
                pass  # (T,1,H,W)
            elif arr.ndim == 3:
                # Could be (T,H,W) -> add channel dim
                arr = arr[:, None, :, :]
            elif arr.ndim == 2:
                # Single map -> (1,1,H,W)
                arr = arr[None, None, :, :]
            else:
                raise ValueError(f"Unsupported field map shape {arr.shape} for {p}")
            # If time dim = 1 and we expect more, repeat
            if arr.shape[0] == 1 and T_expected > 1:
                arr = np.repeat(arr, T_expected, axis=0)
            elif arr.shape[0] != T_expected:
                # Clip to min(T)
                T_use = min(arr.shape[0], T_expected)
                arr = arr[:T_use]
            return torch.from_numpy(arr.astype(np.float32)).to(device)

        Hx = _try_load(path_x)
        Hy = _try_load(path_y)
        if Hx is None or Hy is None:
            # Constant provider; we don't need field grid size here
            return FieldProvider(None, None, const_axis, const_H0, T_expected)
        return FieldProvider(Hx, Hy, const_axis, const_H0, T_expected)

    def sample(self, xy: torch.Tensor, fid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample Hx, Hy at iso coords (B,2). If constant, just return constants.
        When maps are provided with (Hf,Wf), we convert iso coords -> meters ->
        field-pixels using dx_f=Lx/Wf, dy_f=Ly/Hf, then grid_sample.
        """
        if self.const:
            B = xy.shape[0]
            if self.axis.lower() == 'y':
                return (torch.zeros((B, 1), device=xy.device),
                        torch.full((B, 1), float(self.H0), device=xy.device))
            else:
                return (torch.full((B, 1), float(self.H0), device=xy.device),
                        torch.zeros((B, 1), device=xy.device))

        # Use field map spatial size (Hf,Wf)
        Hf, Wf = self.field_H, self.field_W
        dx_f = Lx / Wf
        dy_f = Ly / Hf

        # iso -> meters
        xm = xy[:, 0:1] * s_iso
        ym = xy[:, 1:2] * s_iso
        # meters -> field pixels
        x_px_f = xm / dx_f
        y_px_f = ym / dy_f
        gx = (x_px_f / (Wf - 1)) * 2 - 1
        gy = (y_px_f / (Hf - 1)) * 2 - 1
        grid = torch.stack([gx.squeeze(1), gy.squeeze(1)], dim=1).view(1, 1, -1, 2)

        Hx_out = torch.empty((xy.shape[0], 1), device=xy.device)
        Hy_out = torch.empty((xy.shape[0], 1), device=xy.device)
        uniq = fid.unique()
        for k in uniq.tolist():
            msk = (fid == k)
            if msk.sum() == 0:
                continue
            grid_k = grid[:, :, msk.nonzero(as_tuple=False).squeeze(1), :]
            Hxk = F.grid_sample(self.Hx[k:k+1], grid_k, mode="bilinear", align_corners=True)
            Hyg = F.grid_sample(self.Hy[k:k+1], grid_k, mode="bilinear", align_corners=True)
            Hx_out[msk] = Hxk.view(-1, 1)
            Hy_out[msk] = Hyg.view(-1, 1)
        return Hx_out, Hy_out


class MLP(nn.Module):
    def __init__(self, in_dim, width=160, depth=6, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class Model(nn.Module):
    def __init__(self, with_time: bool, n_frames: int):
        super().__init__()
        in_dim = 3 if with_time else 2
        self.sdf = MLP(in_dim=in_dim, width=160, depth=6, out_dim=1)
        # Learnable physical parameters (dimensionless tildes under iso scale)
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))
        self.gtilde = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1e-6, dtype=torch.float32))  # magnetic pressure scale
        self.P0 = nn.Parameter(torch.zeros(n_frames, dtype=torch.float32))   # per-frame offset
        self.with_time = with_time

    @property
    def gamma_tilde(self):
        return torch.exp(self.log_gamma)

    def forward_s(self, xy: torch.Tensor, t: torch.Tensor | None):
        if self.with_time:
            assert t is not None
            return self.sdf(torch.cat([xy, t], 1))
        return self.sdf(xy)


# =========================
# Autograd helpers
# =========================
def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]


def curvature_from_sdf(s: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g = grad(s, xy)  # (B,2)
    eps = 1e-8
    gnorm = torch.clamp(torch.linalg.norm(g, dim=1, keepdim=True), min=eps)
    nvec = g / gnorm
    dnx = grad(nvec[:, 0:1], xy)
    dny = grad(nvec[:, 1:2], xy)
    kappa = dnx[:, 0:1] + dny[:, 1:2]
    return kappa.squeeze(1), gnorm.squeeze(1)


# =========================
# Dataset builder
# =========================
def make_all_contour_points(frames: List[FrameData]) -> np.ndarray:
    chunks = []
    for idx, fr in enumerate(frames):
        cnt = fr.contour_iso.astype(np.float32)
        t = np.full((cnt.shape[0], 1), fr.tn, np.float32)
        fid = np.full((cnt.shape[0], 1), idx, np.int64)
        chunks.append(np.concatenate([cnt, t, fid], 1))
    return np.concatenate(chunks, 0)  # (N, 4)


def sample_batch(all_cnt: np.ndarray, batch_size: int, device: str):
    idx = rng_np.choice(all_cnt.shape[0], size=min(batch_size, all_cnt.shape[0]), replace=False)
    B = all_cnt[idx]
    xy = torch.from_numpy(B[:, 0:2]).to(device).requires_grad_(True)
    t = torch.from_numpy(B[:, 2:3]).to(device)
    fid = torch.from_numpy(B[:, 3].astype(np.int64)).to(device)
    noise = torch.from_numpy(rng_np.normal(0, 0.01, size=xy.shape).astype(np.float32)).to(device)
    band = (xy + noise).detach().requires_grad_(True)
    return xy, t, fid, band


# =========================
# Area approximation over a grid
# =========================
def approx_area_iso(model: Model, tval: float, N: int, device: str,
                    bounds: Tuple[float,float,float,float] | None = None) -> float:
    # Evaluate s on a grid and take fraction <0 times box area.
    if bounds is None:
        # default square [-1,1]^2 (area = 4)
        xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
        box_area = (xmax - xmin) * (ymax - ymin)
    else:
        xmin, xmax, ymin, ymax = bounds
        box_area = (xmax - xmin) * (ymax - ymin)

    xs = torch.linspace(xmin, xmax, N, device=device)
    ys = torch.linspace(ymin, ymax, N, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    if model.with_time:
        tt = torch.full((xy.shape[0], 1), tval, device=device)
        with torch.no_grad():
            s = model.forward_s(xy, tt)
    else:
        with torch.no_grad():
            s = model.forward_s(xy, None)
    frac_in = (s.reshape(-1) < 0).float().mean()
    return float(frac_in * box_area)


# =========================
# Training
# =========================
def train():
    frames, meta = load_json_iso(JSON_PATH)
    nF = len(frames)
    with_time = (nF > 1)

    # Field provider
    fp = FieldProvider.from_paths_or_const(
        FIELD_PATH_HX, FIELD_PATH_HY, nF, H_img, W_img, DEVICE,
        const_axis=CONST_FIELD_AXIS, const_H0=CONST_FIELD_H0
    )

    model = Model(with_time=with_time, n_frames=nF).to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3)

    all_cnt = make_all_contour_points(frames)
    target_areas = torch.tensor([fr.area_iso for fr in frames], device=DEVICE)

    # Bounds for area box: compute from all contour points
    all_xy = torch.from_numpy(np.concatenate([fr.contour_iso for fr in frames], 0)).to(DEVICE)
    xmin = float(all_xy[:, 0].min().item()) - 0.25
    xmax = float(all_xy[:, 0].max().item()) + 0.25
    ymin = float(all_xy[:, 1].min().item()) - 0.25
    ymax = float(all_xy[:, 1].max().item()) + 0.25
    area_bounds = (xmin, xmax, ymin, ymax)

    for it in range(1, MAX_ITERS + 1):
        xy, t, fid, band = sample_batch(all_cnt, BATCH_CONTOUR, DEVICE)

        # Data: s(contour) -> 0
        s_cnt = model.forward_s(xy, t) if with_time else model.forward_s(xy, None)
        loss_data = (s_cnt ** 2).mean()

        # Eikonal on narrow band
        s_band = model.forward_s(band, t) if with_time else model.forward_s(band, None)
        g_band = grad(s_band, band)
        loss_eik = ((torch.linalg.norm(g_band, dim=1) - 1.0) ** 2).mean()

        # Physics on contour: gamma*kappa - (P0 + g~*y + p_mag) = 0
        kappa, gnorm = curvature_from_sdf(s_cnt, xy)

        # Normal vector at contour
        g = grad(s_cnt, xy)
        gnorm_full = torch.clamp(torch.linalg.norm(g, dim=1, keepdim=True), min=1e-8)
        nvec = g / gnorm_full

        # Sample field
        Hx_s, Hy_s = fp.sample(xy, fid)
        Hn = Hx_s * nvec[:, 0:1] + Hy_s * nvec[:, 1:2]
        p_mag = model.alpha * (Hn ** 2)

        y = xy[:, 1:2]
        P0 = model.P0[fid]
        resid = model.gamma_tilde * kappa.unsqueeze(1) - (P0.unsqueeze(1) + model.gtilde * y + p_mag)
        w_if = torch.exp(-torch.abs(s_cnt) / SIGMA_IF)
        loss_phys = (w_if * resid ** 2).mean()

        # Area regularization over box around data (cheap N)
        loss_area_val = 0.0
        for idx, fr in enumerate(frames):
            a_pred = approx_area_iso(model, fr.tn, GRID_N_AREA, DEVICE, area_bounds)
            loss_area_val += (a_pred - fr.area_iso) ** 2
        loss_area = torch.tensor(loss_area_val / max(1, nF), device=DEVICE)

        loss = W_DATA * loss_data + W_EIK * loss_eik + W_PHYS * loss_phys + W_AREA * loss_area

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            print(f"[{it:05d}] L={float(loss):.4f} data={float(loss_data):.4f} eik={float(loss_eik):.4f} "
                  f"phys={float(loss_phys):.4f} area={float(loss_area):.4f} | "
                  f"gamma~={float(model.gamma_tilde):.5f} g~={float(model.gtilde):.5f} alpha={float(model.alpha):.5e}")

    torch.save({
        "state_dict": model.state_dict(),
        "meta": meta,
        "n_frames": nF,
        "area_bounds": area_bounds,
        "config": {
            "W_DATA": W_DATA, "W_EIK": W_EIK, "W_PHYS": W_PHYS, "W_AREA": W_AREA,
            "SIGMA_IF": SIGMA_IF, "GRID_N_AREA": GRID_N_AREA,
        }
    }, "pinn_inverse_from_json.pt")
    print("Saved: pinn_inverse_from_json.pt")

    # Optional quick preview
    try:
        import matplotlib.pyplot as plt
        N = 256
        xs = torch.linspace(area_bounds[0], area_bounds[1], N, device=DEVICE)
        ys = torch.linspace(area_bounds[2], area_bounds[3], N, device=DEVICE)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        xy_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        if with_time:
            tt = torch.full((xy_grid.shape[0], 1), frames[0].tn, device=DEVICE)
            with torch.no_grad():
                S = model.forward_s(xy_grid, tt).reshape(N, N).cpu().numpy()
        else:
            with torch.no_grad():
                S = model.forward_s(xy_grid, None).reshape(N, N).cpu().numpy()
        plt.figure(figsize=(5, 3))
        plt.contour(X.cpu().numpy(), Y.cpu().numpy(), S, levels=[0.0])
        cnt = frames[0].contour_iso
        plt.plot(cnt[:, 0], cnt[:, 1], '.', ms=1, alpha=0.5)
        plt.gca().set_aspect('equal')
        plt.title('SDF iso(s=0) vs input contour (iso coords)')
        plt.savefig('pinn_iso_contour.png', dpi=200)
        print("Saved: pinn_iso_contour.png")
    except Exception as e:
        print("Preview error:", e)


if __name__ == "__main__":
    train()
