"""
GUI is mainly written by ChatGPT

Angepasst auf deine *aktuellen* Dateien/APIs:
- ConfigStore nutzt .config (nicht .cfg) und load(filepath)
- Logger kommt aus log.py (nicht registry.py)
- Keine projekt-spezifischen Three-Mass-Imports (FitHybrid4, rk4_step, .mat Dateien, etc.)
- EDMDcTrainer/EDMDcModel aus edmdc.py wird verwendet
- Trajektorien/RMSE basieren auf dem synthetischen linearen System aus data_provider.py
- Reload/Signature kompatibel zur registry.py (sha256 über relevante config keys)
"""
from __future__ import annotations

import os
import time
import traceback
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
from flask import request, jsonify
import requests

# --- Projekt-Imports ---
from config_store import ConfigStore
from observables import Observables
from data_provider import DataProvider
from parser import ExpressionParserXOnly
from log import Logger
from edmdc import EDMDcParams, EDMDcTrainer, EDMDcModel

import plotly.io as pio
pio.templates.default = "plotly_dark"

# ===================== Grundsetup =====================
CFG_PATH = os.environ.get("OBS_CFG_PATH", "config.json")
DASH_HOST = os.environ.get("DASH_HOST", "127.0.0.1")
DASH_PORT = int(os.environ.get("DASH_PORT", "8050"))

store = ConfigStore(CFG_PATH)
parser = ExpressionParserXOnly()
provider = DataProvider()

# Sampling time (für Plots)
dT = float(getattr(provider, "dt", 1.0))

MODEL: Optional[EDMDcModel] = None
OBS: Optional[Observables] = None

logger = Logger(level=os.environ.get("OBS_LOG_LEVEL", "INFO"))
use_logger = True  # ausführliches Logging

def log(msg: str) -> None:
    if not use_logger:
        return
    try:
        logger.info(msg)
    except Exception:
        print(msg)


# ===================== Signatur kompatibel zu registry.py =====================
def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _current_signature() -> str:
    # identisch zu ObservableRegistry.signature() (registry.py)
    store.load(CFG_PATH)
    cfg = store.config
    relevant = {
        "include_state": cfg.get("include_state"),
        "include_bias": cfg.get("include_bias"),
        "time_delay": cfg.get("time_delay"),
        "rbf": cfg.get("rbf"),
        "custom_features": cfg.get("custom_features"),
        "gates": cfg.get("gates"),
        "trajectory_horizon": cfg.get("trajectory_horizon"),
        "edmdc_params": cfg.get("edmdc_params"),
        "lift_raw_input": cfg.get("lift_raw_input"),
        "bilinear_zxu": cfg.get("bilinear_zxu"),
    }
    return _sha256_json(relevant)


def build_observables() -> Observables:
    store.load(CFG_PATH)
    cfg = store.config
    return Observables(config=cfg, parser=parser, n_input_features=provider.nx)
def k_metrics(K: np.ndarray, tol_rel: float = 1e-6) -> dict:
    if K.size == 0:
        return {"cond": float("inf"), "eff_rank": 0, "k95": 0}
    lam = np.sort(np.linalg.eigvalsh(K))[::-1]
    lam_pos = lam[lam > 0]
    cond = float(lam_pos[0] / lam_pos[-1]) if lam_pos.size else float("inf")
    thr = tol_rel * lam[0] if lam.size else 0.0
    eff_rank = int(np.sum(lam >= thr))
    if lam.sum() > 0:
        cum = np.cumsum(lam) / lam.sum()
        k95 = int(np.searchsorted(cum, 0.95) + 1)
    else:
        k95 = 0
    return {"cond": cond, "eff_rank": eff_rank, "k95": k95}

# ===================== Training & Artefakte =====================
def fit_model():
    """
    Trainiert EDMDc, berechnet Artefakte (klein!) für die GUI.
    """
    global MODEL, OBS

    log("fit_model: START")
    OBS = build_observables()
    nZ_decl = int(OBS.calculate_output_features())
    log(f"fit_model: observables built (declared nZ={nZ_decl})")

    if nZ_decl < 1:
        log("fit_model: WARN – keine Observables, Training wird übersprungen")
        return None, {"dt": dT, "nZ": 0, "spec": {"eig": [], "rho": 0.0, "outside": 0}, "K_stats": {}}

    cfg = store.config
    edmdc_cfg = cfg.get("edmdc_params", {"lambda_dyn": 1e-6, "lambda_dec": 1e-6})
    params = EDMDcParams(
        lambda_dyn=float(edmdc_cfg.get("lambda_dyn", 1e-6)),
        lambda_dec=float(edmdc_cfg.get("lambda_dec", 1e-6)),
    )
    trainer = EDMDcTrainer(params)

    log("fit_model: TRAINING START")
    MODEL = trainer.fit(X=provider.X, Y=provider.Y, U=provider.U, observables=OBS)
    log("fit_model: TRAINING DONE")

    # Phi & K
    Phi = np.asarray(OBS.transform(provider.X), dtype=float)  # (N_eff, nZ)
    if Phi.ndim != 2 or Phi.shape[0] < 1:
        log(f"fit_model: WARN – Phi invalid: {Phi.shape}")
        return MODEL, {"dt": dT, "nZ": 0, "spec": {"eig": [], "rho": 0.0, "outside": 0}, "K_stats": {}}

    K_full = (Phi.T @ Phi) / max(1, Phi.shape[0])
    log(f"fit_model: Phi computed -> K_full shape={K_full.shape}")

    # Koopman-Spektrum
    eig = np.linalg.eigvals(MODEL.A)
    rho = float(np.max(np.abs(eig))) if eig.size else 0.0
    outside = int(np.sum(np.abs(eig) > 1 + 1e-9))
    eig_pairs = [[float(z.real), float(z.imag)] for z in eig]

    nZ = int(Phi.shape[1])
    art = {
        "dt": float(dT),
        "nZ": nZ,
        "spec": {"eig": eig_pairs, "rho": rho, "outside": outside},
        "K_stats": k_metrics(K_full),
    }

    if nZ <= 25:
        art["K_thumb"] = K_full.tolist()

    log(f"fit_model: DONE (nZ={nZ}, rho={rho:.4f}, outside={outside})")
    return MODEL, art

def normalized_rmse_curve_over_trajs(X_truth_list, X_hat_list):
    """
    Normalisierte RMSE-Kurve über mehrere Trajektorien:
        RMSE(k) = 100 * sqrt( sum_{i} sum_{j=0}^{k-1} ||x̂_{i,j} - x_{i,j}||^2
                              / sum_{i} sum_{j=0}^{k-1} ||x_{i,j}||^2 )
    """
    assert len(X_truth_list) == len(X_hat_list)
    if not X_truth_list:
        return np.array([]), np.array([])

    T_min = min(X.shape[0] for X in X_truth_list)
    ks = np.arange(1, T_min + 1, dtype=int)
    errs = np.zeros_like(ks, dtype=float)

    for idx, k in enumerate(ks):
        num = 0.0
        den = 0.0
        for Xt, Xh in zip(X_truth_list, X_hat_list):
            diff = Xh[:k, :] - Xt[:k, :]
            num += np.sum(diff**2)
            den += np.sum(Xt[:k, :]**2)
        errs[idx] = 100.0 * np.sqrt(num / den) if den > 0 else np.nan

    return ks, errs


def spectrum_fig_from_eigs(eig_curr: np.ndarray, eig_prev: Optional[np.ndarray] = None) -> go.Figure:
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 400)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines", name="unit circle",
        line=dict(color="black", dash="dot")
    ))
    if eig_prev is not None and eig_prev.size:
        fig.add_trace(go.Scatter(
            x=eig_prev.real, y=eig_prev.imag,
            mode="markers", name="previous",
            marker=dict(size=8, symbol="x")
        ))
    if eig_curr is not None and eig_curr.size:
        fig.add_trace(go.Scatter(
            x=eig_curr.real, y=eig_curr.imag,
            mode="markers", name="current",
            marker=dict(size=8, symbol="circle")
        ))
    fig.update_layout(
        autosize=True,
        xaxis=dict(scaleanchor="y", scaleratio=1, zeroline=True, showgrid=True, title="Re(λ)"),
        yaxis=dict(zeroline=True, showgrid=True, title="Im(λ)"),
        margin=dict(l=30, r=10, t=30, b=30),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15)
    )
    return fig

# ===================== Dash App =====================
app = Dash(__name__)
app.title = "EDMDc – Visualizer"
server = app.server

# --- Reload-API (Terminal-Notify) ---
RELOAD_TS = 0.0
LAST_SIGNATURE = None

@server.route("/api/reload", methods=["POST"])
def api_reload():
    global RELOAD_TS, LAST_SIGNATURE
    data = request.get_json(silent=True) or {}
    LAST_SIGNATURE = data.get("signature")
    RELOAD_TS = time.time()
    return ("", 204)

@server.route("/api/reload-flag", methods=["GET"])
def api_reload_flag():
    return jsonify({"ts": RELOAD_TS, "signature": LAST_SIGNATURE})


# Busy-Flag nur intern (zusätzlich gibt es ein UI-Busy in dcc.Store)
IS_FITTING = False
RUN_COUNT = 0  # nur fürs Log

app.layout = html.Div([
    # Stores
    dcc.Store(id="store-artifacts"),
    dcc.Store(id="store-prev-spec"),
    dcc.Store(id="store-prev-RMSE"),
    dcc.Store(id="store-prev-trajs"),
    dcc.Store(id="store-feature-count"),
    dcc.Store(id="store-signature", data=_current_signature()),
    dcc.Store(id="store-refresh", data=0),
    dcc.Store(id="store-busy", data=False),
    dcc.Store(id="store-runreq", data=0),
    dcc.Store(id="store-pending-sig", data=None),

    # UI
    dcc.Interval(id="poll", interval=3000, n_intervals=0),
    html.Div([
        html.Button("Fit", id="btn-fit", n_clicks=0, disabled=False),
        html.Span(id="reload-status",
                  style={"marginLeft": "16px", "color": "#2d6a4f", "fontWeight": 600})
    ], style={"margin": "10px 0"}),

    html.Div([
        # LEFT: trajectories
        html.Div(id="col-traj", style={
            "flex": "0 0 60%",
            "border": "1px solid #ddd",
            "padding": "8px",
            "height": "82vh",
            "overflowY": "auto",
            "borderRadius": "8px",
            "background": "#fff",
            "minWidth": "420px"
        }),

        # MIDDLE: RMSE + Spectrum (2x1 bleibt)
        html.Div([
            html.Div(id="box-rmse", style={
                "height": "40vh",
                "marginBottom": "8px",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "6px",
                "background": "#fff"
            }),
            html.Div(id="box-kprev", style={
                "height": "40vh",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "6px",
                "background": "#fff"
            }),
        ], style={"flex": "0 0 20%", "margin": "0 6px", "minWidth": "260px"}),

        # RIGHT: Observables full height (kein Grid mehr)
        html.Div([
            html.Div(id="box-phi", style={
                "height": "82vh",            # <- volle Höhe wie Traj-Spalte
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "padding": "10px",
                "whiteSpace": "pre-wrap",
                "fontFamily": "Consolas, monospace",
                "background": "#fff",
                "overflowY": "auto",
            }),
        ], style={"flex": "0 0 20%", "minWidth": "260px"}),
    ], style={"display": "flex", "gap": "6px", "minHeight": "0"})
], style={"padding": "10px 14px", "fontFamily": "Inter, Segoe UI, Roboto, sans-serif",
          "background": "#f6f7fb"})

log("APP: layout ready")


# ============= Helpers für Rollouts =============
def _simulate_truth(A: np.ndarray, B: np.ndarray, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Simuliert diskret: x_{k+1} = A x_k + B u_k (ohne Noise)
    Gibt X mit Länge (T+1,nX) zurück.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    T = int(U.shape[0])
    nx = int(A.shape[0])
    X = np.zeros((T + 1, nx), dtype=float)
    x = x0.copy()
    X[0] = x
    for k in range(T):
        u = U[k].reshape(-1)
        x = A @ x + (B @ u if B.size else 0.0)
        X[k + 1] = x
    return X


def _rollout_one(model: EDMDcModel, *, horizon: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Erstellt eine Truth-Trajektorie + Prediction + U für den Plot.
    Berücksichtigt Time-Delay, falls aktiv.
    Rückgabe:
      Xt_plot: (T,nX) truth aligned
      Xh:      (T,nX) prediction aligned
      Ut:      (T,nU)
    """
    cfg: Dict[str, Any] = getattr(model.observables, "config", {}) or {}
    td = cfg.get("time_delay", {}) or {}
    td_enabled = bool(td.get("enabled", False))
    n_delays = int(td.get("n_delays", 0))
    tau = int(td.get("tau", 1))
    delay_width = (n_delays * tau) if td_enabled else 0

    nx = int(provider.nx)
    nU = int(provider.nU)

    # Inputs
    Ut = rng.normal(scale=1.0, size=(horizon + delay_width + 1, nU)).astype(float)

    # Startzustand
    x0 = rng.normal(scale=0.5, size=(nx,)).astype(float)

    # Truth simulieren (Länge: (horizon+delay_width+2))
    X_truth_full = _simulate_truth(provider.A_true, provider.B_true, x0, Ut)

    # Prediction
    if delay_width > 0:
        x_hist = X_truth_full[:delay_width + 1, :]                 # x_0 .. x_delay_width
        U_seq = Ut[delay_width:delay_width + horizon, :]           # u_delay_width .. u_delay_width+horizon-1
        Xh = model.predict_multi_step(x_hist, U_seq)               # predicts x_{delay_width+1} .. x_{delay_width+horizon}
        Xt_plot = X_truth_full[delay_width + 1:delay_width + 1 + horizon, :]
        Ut_plot = U_seq
    else:
        U_seq = Ut[:horizon, :]
        Xh = model.predict_multi_step(X_truth_full[0, :], U_seq)   # predicts x_1..x_horizon
        Xt_plot = X_truth_full[1:horizon + 1, :]
        Ut_plot = U_seq

    return Xt_plot, np.asarray(Xh, dtype=float), Ut_plot


def _auto_fit(art_prev, prev_spec, rmse_prev, nZ_prev, prev_trajs, reason: str):
    global IS_FITTING, RUN_COUNT
    if IS_FITTING:
        log(f"auto_fit: BUSY -> skip (reason={reason})")
        return art_prev or {}, prev_spec, rmse_prev, nZ_prev, prev_trajs

    IS_FITTING = True
    RUN_COUNT += 1
    run_id = RUN_COUNT

    log(f"[RUN #{run_id}] ===== START ({reason}) =====")
    try:
        model, art = fit_model()
        if model is None or not art.get("nZ", 0):
            log(f"[RUN #{run_id}] NO MODEL -> abort")
            return art, prev_spec, rmse_prev, nZ_prev, prev_trajs

        store.load(CFG_PATH)
        T = int(store.config.get("trajectory_horizon", 100))
        T = max(5, min(T, 500))  # GUI-sicher

        # Mehrere Rollouts
        rng = np.random.default_rng(int(time.time()) % (2**32 - 1))
        n_trajs = int(os.environ.get("OBS_GUI_TRAJS", "6"))
        n_trajs = max(1, min(n_trajs, 20))

        trajs_cur = []
        truth_list = []
        hat_list = []

        for _ in range(n_trajs):
            Xt_plot, Xh, Ut_plot = _rollout_one(model, horizon=T, rng=rng)

            t = np.arange(T, dtype=float) * float(dT)

            traj = {
                "t": t.tolist(),
                "tt": t.tolist(),
                "u": Ut_plot.reshape(T, -1)[:, 0].tolist() if Ut_plot.size else [0.0] * T,
            }
            nx_cur = Xt_plot.shape[1]
            for j in range(nx_cur):
                traj[f"x{j+1}_truth"] = Xt_plot[:, j].tolist()
                traj[f"x{j+1}_hat"] = Xh[:, j].tolist()

            trajs_cur.append(traj)
            truth_list.append(Xt_plot)
            hat_list.append(Xh)

        # RMSE über alle Testtrajektorien
        ks, errs = normalized_rmse_curve_over_trajs(truth_list, hat_list)
        art["rmse_curve"] = {"ks": ks.tolist(), "errs": errs.tolist()}
        art["trajs"] = trajs_cur

        # Rev-Stamp erzwingen
        art["rev"] = time.time()

        log(f"[RUN #{run_id}] PUBLISH: artifacts prepared (nZ={art.get('nZ')}, trajs={len(trajs_cur)})")
        log(f"[RUN #{run_id}] ===== COMPLETE =====")
        return art, art.get("spec"), art.get("rmse_curve"), int(art.get("nZ", 0)), trajs_cur
    finally:
        IS_FITTING = False
        log(f"[RUN #{run_id}] RELEASE BUSY")


def _which_triggered():
    try:
        trig = callback_context.triggered
    except Exception:
        try:
            from dash import ctx
            return ctx.triggered_id
        except Exception:
            return None
    if not trig:
        return None
    prop = trig[0].get("prop_id", "")
    return prop.split(".")[0] if "." in prop else prop


# ===================== Callbacks =====================

# A) Button sofort an Busy koppeln
@app.callback(
    Output("btn-fit", "disabled"),
    Input("store-busy", "data"),
    prevent_initial_call=False
)
def disable_button(busy):
    return bool(busy)

# B) Kickoff: Button/Poll stoßen Fit an
@app.callback(
    Output("store-runreq", "data"),
    Output("store-busy", "data", allow_duplicate=True),
    Output("store-pending-sig", "data"),
    Output("reload-status", "children", allow_duplicate=True),
    Input("btn-fit", "n_clicks"),
    Input("poll", "n_intervals"),
    State("store-busy", "data"),
    State("store-signature", "data"),
    State("store-runreq", "data"),
    prevent_initial_call="initial_duplicate"   # <- FIX
)
def kickoff_fit(n_clicks, _n_intervals, busy, prev_sig, runreq):
    which = _which_triggered()
    ts = time.strftime("%H:%M:%S")

    if busy:
        return no_update, no_update, no_update, no_update

    if which == "btn-fit":
        new_runreq = int(runreq or 0) + 1
        return new_runreq, True, None, f"Manual fit started @ {ts} | Signature: {prev_sig or '-'}"

    if which == "poll":
        try:
            r = requests.get(f"http://{DASH_HOST}:{DASH_PORT}/api/reload-flag", timeout=1.0)
            info = r.json() if r.ok else {}
            sig_remote = info.get("signature")
        except Exception:
            sig_remote = None

        if sig_remote and sig_remote != prev_sig:
            new_runreq = int(runreq or 0) + 1
            return new_runreq, True, sig_remote, f"Auto-reload started @ {ts} | New signature: {sig_remote}"

    return no_update, no_update, no_update, no_update


# C) Do-Fit: läuft genau einmal pro Kickoff und published erst am Ende
@app.callback(
    Output("store-signature", "data"),
    Output("reload-status", "children", allow_duplicate=True),
    Output("store-artifacts", "data"),
    Output("store-prev-spec", "data"),
    Output("store-prev-RMSE", "data"),
    Output("store-feature-count", "data"),
    Output("store-prev-trajs", "data"),
    Output("store-refresh", "data"),
    Output("store-busy", "data", allow_duplicate=True),
    Input("store-runreq", "data"),
    State("store-signature", "data"),
    State("store-artifacts", "data"),
    State("store-prev-spec", "data"),
    State("store-prev-RMSE", "data"),
    State("store-feature-count", "data"),
    State("store-prev-trajs", "data"),
    State("store-refresh", "data"),
    State("store-pending-sig", "data"),
    prevent_initial_call=True
)
def do_fit(runreq, prev_sig, art_prev, prev_spec, rmse_prev, nZ_prev, prev_trajs, refresh_prev, pending_sig):
    if not runreq:
        return (prev_sig, no_update, no_update, no_update, no_update, no_update,
                no_update, int(refresh_prev or 0), False)

    reason = "auto-reload" if pending_sig else "manual"
    try:
        art, spec_curr, rmse_curr, nZ_curr, trajs_curr = _auto_fit(
            art_prev, prev_spec, rmse_prev, nZ_prev, prev_trajs, reason
        )

        prev_spec_out = (art_prev or {}).get("spec", prev_spec)
        prev_rmse_out = (art_prev or {}).get("rmse_curve", rmse_prev)
        prev_trajs_out = (art_prev or {}).get("trajs", prev_trajs)

        if isinstance(art, dict):
            art["spec_prev"] = prev_spec_out

        new_refresh = int(refresh_prev or 0) + 1
        ts = time.strftime("%H:%M:%S")

        new_sig = pending_sig if pending_sig else _current_signature()
        label = (f"Auto-reload @ {ts} | Signature: {new_sig}"
                 if pending_sig else f"Manual fit @ {ts} | Signature: {new_sig or '-'}")

        log(f"do_fit: returning busy=False, refresh={new_refresh}")
        return (new_sig, label, art, prev_spec_out, prev_rmse_out, int(nZ_curr or 0),
                prev_trajs_out, new_refresh, False)

    except Exception as e:
        log(f"do_fit ERROR: {e}\n{traceback.format_exc()}")
        fallback_refresh = int(refresh_prev or 0) + 1
        label = f"Fit failed: {e}"
        return (prev_sig, label, art_prev, prev_spec, rmse_prev, int(nZ_prev or 0),
                prev_trajs, fallback_refresh, False)


def _single_state_fig(t, y_truth, y_hat, y_prev, y_lin, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y_truth, name=f"{name} truth", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=y_hat, name=f"{name} EDMDc", line=dict(width=2, dash="dash")))
    if y_lin is not None:
        fig.add_trace(go.Scatter(x=t, y=y_lin, name=f"{name} linear", line=dict(width=2, dash="dot")))
    if y_prev is not None:
        fig.add_trace(go.Scatter(x=t, y=y_prev, name=f"{name} prev", line=dict(width=2, dash="longdash")))
    fig.update_xaxes(title="t")
    fig.update_yaxes(title=name)
    fig.update_layout(
        width=380, height=260,
        margin=dict(l=20, r=12, t=8, b=30),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25, yanchor="top",
                    bgcolor="rgba(255,255,255,0.65)")
    )
    return fig


# D) Trajectories – nur rendern, wenn nicht busy
@app.callback(
    Output("col-traj", "children"),
    Input("store-refresh", "data"),
    Input("store-busy", "data"),
    State("store-artifacts", "data"),
    State("store-prev-trajs", "data"),
    prevent_initial_call=True
)
def render_trajs(refresh, busy, art, prev_trajs):
    log(f"render_trajs: ENTER (refresh={refresh}, busy={busy})")
    if busy:
        return [html.Div("Training läuft…", style={"padding": "12px"})]

    if not art or "trajs" not in art or not art["trajs"]:
        log("render_trajs: no trajs -> placeholder")
        return [html.Div("Noch kein Fit.", style={"padding": "12px"})]

    trajs_cur = art["trajs"]
    items = []

    def _nx_from_payload(pdict: dict) -> int:
        idx = 0
        for k in pdict.keys():
            if k.endswith("_truth") and k.startswith("x"):
                try:
                    idx = max(idx, int(k[1:].split("_")[0]))
                except Exception:
                    pass
        return idx

    for idx, p in enumerate(trajs_cur):
        t = np.array(p["t"], dtype=float)
        nx = _nx_from_payload(p)
        row_plots = []

        pprev = prev_trajs[idx] if (prev_trajs is not None and idx < len(prev_trajs)) else None
        for j in range(1, nx + 1):
            y_truth = np.array(p[f"x{j}_truth"], dtype=float)
            y_hat = np.array(p[f"x{j}_hat"], dtype=float)
            y_prev = np.array(pprev.get(f"x{j}_hat"), dtype=float) if (pprev and f"x{j}_hat" in pprev) else None
            y_lin = np.array(p.get(f"x{j}_lin"), dtype=float) if (f"x{j}_lin" in p) else None

            fig = _single_state_fig(t, y_truth, y_hat, y_prev, y_lin, name=f"x{j}")

            row_plots.append(
                html.Div(
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    style={"flex": "0 0 400px", "minWidth": "400px", "maxWidth": "400px"}
                )
            )

        # Stellgröße u separat
        u = np.array(p.get("u", np.zeros_like(t)), dtype=float)
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=t, y=u, name="u(t)", line=dict(width=2)))
        fig_u.update_xaxes(title="t")
        fig_u.update_yaxes(title="u")
        fig_u.update_layout(width=380, height=260, margin=dict(l=20, r=12, t=8, b=30))
        row_plots.append(
            html.Div(
                dcc.Graph(figure=fig_u, config={"displayModeBar": False}),
                style={"flex": "0 0 400px", "minWidth": "400px", "maxWidth": "400px"}
            )
        )

        row = html.Div(
            row_plots,
            style={
                "display": "flex", "flexWrap": "nowrap", "gap": "8px",
                "overflowX": "auto", "padding": "4px 2px"
            }
        )
        items.append(html.Div([row], style={"marginBottom": "12px"}))

    log("render_trajs: DONE")
    return items


# E) RMSE – nur rendern, wenn nicht busy
@app.callback(
    Output("box-rmse", "children"),
    Input("store-refresh", "data"),
    Input("store-busy", "data"),
    State("store-artifacts", "data"),
    State("store-prev-RMSE", "data"),
    prevent_initial_call=True
)
def render_rmse(refresh, busy, art, rmse_prev):
    log(f"render_rmse: ENTER (refresh={refresh}, busy={busy})")
    if busy:
        return html.Div("Training läuft…", style={"padding": "12px"})
    cur = art.get("rmse_curve", {"ks": [], "errs": []}) if art else {"ks": [], "errs": []}
    ks, errs = np.array(cur["ks"]), np.array(cur["errs"])
    fig = go.Figure()
    if ks.size and errs.size:
        fig.add_trace(go.Scatter(x=ks, y=errs, mode="lines+markers", name="current", line=dict(width=3)))
    if rmse_prev is not None:
        ks_p = np.array(rmse_prev.get("ks", []))
        er_p = np.array(rmse_prev.get("errs", []))
        if ks_p.size and er_p.size:
            fig.add_trace(go.Scatter(x=ks_p, y=er_p, mode="lines", name="previous", line=dict(width=2, dash="dash")))

    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k-step", yaxis_title="RMSE [%]",
                      legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                                  bgcolor="rgba(255,255,255,0.6)"))
    log("render_rmse: DONE")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})

# G) Koopman-Spektrum – nur rendern, wenn nicht busy
@app.callback(
    Output("box-kprev", "children"),
    Input("store-refresh", "data"),
    Input("store-busy", "data"),
    State("store-artifacts", "data"),
    prevent_initial_call=True
)
def render_koopman_spectrum(refresh, busy, art):
    log(f"render_koopman_spectrum: ENTER (refresh={refresh}, busy={busy})")
    if busy:
        return html.Div("Training läuft…", style={"padding": "12px"})
    if not art or "spec" not in art:
        log("render_koopman_spectrum: no spec -> placeholder")
        return html.Div("Noch kein Fit.", style={"padding": "12px"})
    spec = art["spec"]
    prev_spec = art.get("spec_prev")
    eig_curr = np.array([c[0] + 1j * c[1] for c in spec.get("eig", [])], dtype=complex)
    eig_prev = None
    if prev_spec and "eig" in prev_spec:
        eig_prev = np.array([c[0] + 1j * c[1] for c in prev_spec["eig"]], dtype=complex)
    fig = spectrum_fig_from_eigs(eig_curr, eig_prev)
    footer = html.Div(
        f"ρ(A) = {spec.get('rho', 0.0):.4f}   |   outside unit circle: {spec.get('outside', 0)}",
        style={"fontSize": "12px", "padding": "4px 2px", "color": "#334"}
    )
    log("render_koopman_spectrum: DONE")
    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False},
                               style={"height": "100%", "width": "100%"}), footer],
                    style={"height": "100%", "display": "flex", "flexDirection": "column"})


# H) Phi-Liste – nur rendern, wenn nicht busy
@app.callback(
    Output("box-phi", "children"),
    Input("store-refresh", "data"),
    Input("store-busy", "data"),
    State("store-artifacts", "data"),
    prevent_initial_call=True
)
def render_phi(refresh, busy, art):
    log(f"render_phi: ENTER (refresh={refresh}, busy={busy})")
    if busy:
        return html.Div("Training läuft…", style={"padding": "12px"})
    try:
        if OBS is None or int(OBS.calculate_output_features()) == 0:
            log("render_phi: no observables")
            return html.Div("Keine Observables definiert.", style={"padding": "12px"})

        cfg = OBS.config

        include_state = bool(cfg.get("include_state", True))
        include_bias = bool(cfg.get("include_bias", False))

        td = cfg.get("time_delay", {}) or {}
        td_enabled = bool(td.get("enabled", False))
        apply_on_x = bool(td.get("apply_on_x", True))
        n_delays = int(td.get("n_delays", 0))
        tau = int(td.get("tau", 1))

        # Base labels (ohne Gates, ohne z-delay)
        labels_base = []

        # 1) state (+ optional x-delay)
        if include_state:
            if td_enabled and apply_on_x and n_delays > 0:
                for d in range(0, n_delays + 1):
                    lag = d * tau
                    suf = "" if lag == 0 else f"(t-{lag})"
                    for i in range(OBS.n_input_features):
                        labels_base.append(f"x{i+1}{suf}")
            else:
                for i in range(OBS.n_input_features):
                    labels_base.append(f"x{i+1}")

        # 2) rbf
        rbf = cfg.get("rbf", {}) or {}
        if bool(rbf.get("enabled", False)) and int(rbf.get("n_centers", 0)) > 0:
            for i in range(int(rbf.get("n_centers", 0))):
                labels_base.append(f"rbf_{i}")

        # 3) custom
        custom = cfg.get("custom_features", []) or []
        for i, expr in enumerate(custom):
            labels_base.append(f"$${str(expr)}$$ : c_{i}")

        # 4) bias
        if include_bias:
            labels_base.append("1")

        # 5) gates
        gcfg = cfg.get("gates", {}) or {}
        gates_enabled = bool(gcfg.get("enabled", False))
        gates_as_features = bool(gcfg.get("gates_as_features", False))
        gates = gcfg.get("items", []) or []

        def _short_expr(expr, n=26):
            s = str(expr).strip().replace("\n", " ")
            return s if len(s) <= n else (s[:n-1] + "…")

        labels = list(labels_base)

        if gates_enabled and gates and labels_base:
            # phi + phi*g1 + ... (gated copies)
            for gi, gexpr in enumerate(gates, start=1):
                gtag = f"g{gi}({_short_expr(gexpr)})"
                labels.extend([f"{lab} · {gtag}" for lab in labels_base])

            if gates_as_features:
                for gi, gexpr in enumerate(gates, start=1):
                    labels.append(f"g{gi}({_short_expr(gexpr)})")

        # 6) z-delay (wenn td_enabled && !apply_on_x)
        if td_enabled and (not apply_on_x) and n_delays > 0 and labels:
            labels_nodelay = labels
            labels = []
            for d in range(0, n_delays + 1):
                lag = d * tau
                suf = "" if lag == 0 else f"(t-{lag})"
                labels.extend([f"{lab}{'' if suf=='' else ' ' + suf}" for lab in labels_nodelay])

        expected = int(OBS.calculate_output_features())
        got = len(labels)
        warn_div = None
        if expected != got:
            msg = (f"Achtung: #Observables ungleich! berechnet={expected}, gerendert={got}. "
                   f"Prüfe include_state / include_bias / time_delay / rbf / custom_features / gates.")
            log("render_phi MISMATCH: " + msg)
            warn_div = html.Div(msg, style={
                "color": "#b00020", "background": "#fdecea",
                "border": "1px solid #f5c2c7", "padding": "6px 8px",
                "borderRadius": "6px", "marginBottom": "6px",
                "fontWeight": 600
            })

        md_text = "\n".join(f"{i+1}. {lab}" for i, lab in enumerate(labels))
        body = dcc.Markdown(md_text, mathjax=True, style={"fontSize": "16px", "fontFamily": "serif"})

        log(f"render_phi: DONE (labels={got}, expected={expected})")
        return html.Div([warn_div, body]) if warn_div is not None else body

    except Exception as ex:
        log(f"render_phi ERROR: {type(ex).__name__}: {ex}")
        return html.Div(f"Labels nicht verfügbar ({ex})", style={"fontSize": "16px"})


# ===================== Main =====================
if __name__ == "__main__":
    log(f"APP: run server (debug=True, reloader=False, host={DASH_HOST}, port={DASH_PORT})")
    app.run(debug=True, use_reloader=False, host=DASH_HOST, port=DASH_PORT)
