from __future__ import annotations

from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from agent.batches import load_case_a_batches, resolve_case_a_batches_path
import pandas as pd
import streamlit_shadcn_ui as ui

from agent.graph import run_agent, generate_snapshot_report
from agent.golden_profile import build_case_a_golden_profile
from agent.monitoring import evaluate_stream, evaluate_stream_dtw
from agent.timeseries import load_case_a_timeseries, phase_segments, resolve_case_a_timeseries_path
from agent.driver_analysis import analyze_batch_against_golden_profile
from agent.multivariate import score_isolation_forest


st.set_page_config(page_title="Golden Batch Detective", page_icon="🔍", layout="wide")


def _check_password() -> bool:
    if st.session_state.get("_authenticated"):
        return True

    expected = st.secrets.get("auth", {}).get("password", "")
    if not expected:
        return True  # No password configured → open access (local dev)

    st.title("Golden Batch Detective 🔍")
    with st.form("login_form"):
        pw = st.text_input("Passwort", type="password")
        if st.form_submit_button("Anmelden"):
            if pw == expected:
                st.session_state["_authenticated"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    return False


if not _check_password():
    st.stop()

st.title("Golden Batch Detective 🔍")

st.markdown("""
<style>
/* ── Warning / Early-Warning box ── */
.gbd-box         { border-radius:6px; padding:10px 14px; box-sizing:border-box;
                   height:90px; overflow-y:auto; }
.gbd-box-wait    { background:rgba(148,163,184,0.08); border-left:3px solid rgba(148,163,184,0.5); }
.gbd-box-ok      { background:rgba(34,197,94,0.10);  border-left:3px solid #22c55e;
                   display:flex; align-items:center; }
.gbd-box-warn    { background:rgba(245,158,11,0.10); border-left:3px solid #f59e0b; }
.gbd-box-crit    { background:rgba(239,68,68,0.10);  border-left:3px solid #ef4444; }
.gbd-label-ok    { font-weight:700; font-size:0.9rem; color:#22c55e; }
.gbd-label-warn  { font-weight:700; font-size:0.9rem; color:#f59e0b; }
.gbd-label-crit  { font-weight:700; font-size:0.9rem; color:#ef4444; }
.gbd-subtitle    { font-weight:400; color:#9ca3af; margin-left:8px; font-size:0.82rem; }
.gbd-vars        { margin-top:5px; font-size:0.82rem; opacity:0.85; }
.gbd-topz        { margin-top:3px; font-size:0.8rem; color:#9ca3af; }
/* ── Batch status card ── */
.gbd-status-card { border-radius:14px; padding:18px 24px; margin-bottom:16px;
                   display:flex; align-items:center; gap:16px; }
.gbd-status-ok   { background:rgba(34,197,94,0.08);  border:2px solid rgba(34,197,94,0.35); }
.gbd-status-bad  { background:rgba(239,68,68,0.08);  border:2px solid rgba(239,68,68,0.35); }
.gbd-status-unk  { background:rgba(245,158,11,0.08); border:2px solid rgba(245,158,11,0.35); }
.gbd-status-lbl-ok  { font-size:1.6rem; font-weight:700; color:#22c55e; line-height:1.2; }
.gbd-status-lbl-bad { font-size:1.6rem; font-weight:700; color:#ef4444; line-height:1.2; }
.gbd-status-lbl-unk { font-size:1.6rem; font-weight:700; color:#f59e0b; line-height:1.2; }
.gbd-status-sub  { font-size:1rem; margin-top:4px; opacity:0.55; }
/* ── AI panel ── */
.gbd-ai-header   { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.gbd-ai-title    { font-weight:700; font-size:1.05rem; color:#818cf8; }
.gbd-ai-ph       { border:1.5px dashed rgba(148,163,184,0.35); border-radius:10px;
                   padding:24px 20px; text-align:center; opacity:0.55; }
/* ── Metric card (dark-mode-safe replacement for ui.metric_card) ── */
.gbd-metric      { background:rgba(148,163,184,0.07); border:1px solid rgba(148,163,184,0.14);
                   border-radius:10px; padding:14px 16px; text-align:left; }
.gbd-metric-title{ font-size:0.78rem; color:#9ca3af; margin-bottom:6px;
                   text-transform:uppercase; letter-spacing:0.04em; }
.gbd-metric-val  { font-size:1.45rem; font-weight:700; }
/* ── Native button tweaks (st.button fills full column width) ── */
div[data-testid="stButton"] > button { width:100%; }
/* ── Batch list container: transparent background in dark mode ── */
div[data-testid="stVerticalBlockBorderWrapper"] { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

repo_root = Path(__file__).resolve().parents[2]

try:
    batches_path = resolve_case_a_batches_path(repo_root)
    batches_by_id = load_case_a_batches(batches_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load caseA_batches.csv: {exc}")
    batches_by_id = {}

try:
    ts_path = resolve_case_a_timeseries_path(repo_root)
    all_points = load_case_a_timeseries(ts_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load caseA_timeseries.csv: {exc}")
    all_points = []

tab_ov, tab_gp, tab_mon, tab_rep = st.tabs(
    ["Batch-Übersicht", "Golden Profile", "Live Monitoring", "Report"]
)


@st.cache_data(show_spinner=False)
def _cached_golden_profile() -> dict:
    return build_case_a_golden_profile(repo_root, t_pct_step=5)


def _profile_rows_for_variable(profile: dict, variable: str) -> list[dict]:
    rows = [r for r in (profile.get("rows") or []) if r.get("variable") == variable]
    # Sort by phase then bucket
    rows.sort(key=lambda r: (str(r.get("phase")), int(r.get("t_pct_bucket", 0))))
    return rows


def _batch_points(batch_id: str):
    pts = [p for p in all_points if p.batch_id == batch_id]
    pts.sort(key=lambda p: p.t_pct)
    return pts


_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#FF9DA6"]
_ALL_PHASES = sorted({str(p.phase) for p in all_points} if all_points else set())
_PHASE_COLOR_MAP: dict[str, str] = {ph: _PALETTE[i % len(_PALETTE)] for i, ph in enumerate(_ALL_PHASES)}


def _phase_color(phase: str) -> str:
    return _PHASE_COLOR_MAP.get(phase, "rgba(160,160,160,0.2)")


def _deviation_color(max_abs_z: float) -> str:
    if max_abs_z >= 2.5:
        return "#E45756"
    if max_abs_z >= 1.2:
        return "#EECA3B"
    return "#54A24B"


def _deviation_chart(batch_id: str) -> go.Figure | None:
    pts = _batch_points(batch_id)
    if not pts:
        return None
    try:
        profile = _cached_golden_profile()
        evals = evaluate_stream(pts, golden_profile=profile, window_size=5, t_pct_step=5, z_threshold=2.0)
    except Exception:  # noqa: BLE001
        return None

    t_vals = [e.t_pct for e in evals]
    max_z = [max((abs(z) for z in e.z_scores.values()), default=0.0) for e in evals]
    marker_colors = [_deviation_color(z) for z in max_z]

    def _hover_text(e) -> str:
        flagged = sorted(
            [(v, e.z_scores[v]) for v, outside in e.flags.items() if outside and v in e.z_scores],
            key=lambda kv: abs(kv[1]), reverse=True,
        )
        if not flagged:
            return "Keine Auffälligkeiten"
        lines = []
        for v, z in flagged:
            direction = "↑ zu hoch" if z > 0 else "↓ zu niedrig"
            lines.append(f"{v}: {direction}")
        return "<br>".join(lines)

    hover_texts = [_hover_text(e) for e in evals]

    fig = go.Figure()
    segs = phase_segments(pts)
    _add_phase_bands(fig, segs)

    seen_phases: set[str] = set()
    phases_present = [str(s.phase) for s in segs if str(s.phase) and str(s.phase) not in seen_phases and not seen_phases.add(str(s.phase))]  # type: ignore[func-returns-value]
    for ph in phases_present:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=_phase_color(ph)),
            name=f"Phase: {ph}", showlegend=True, hoverinfo="skip",
        ))

    y_max = max(max_z + [2.5])
    fig.add_hrect(y0=2.0, y1=y_max * 1.05, fillcolor="rgba(228,87,86,0.07)", line_width=0)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#E45756",
                  annotation_text="Warnschwelle", annotation_position="top right")

    fig.add_trace(go.Scatter(
        x=t_vals, y=max_z,
        mode="lines+markers",
        line=dict(color="rgba(100,100,100,0.4)", width=1.5),
        marker=dict(color=marker_colors, size=7),
        name="Abweichung",
        text=hover_texts,
        hovertemplate="<b>Zeitfortschritt: %{x:.1f} %</b><br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Zeitfortschritt (%)",
        yaxis_title="Stärke der Abweichung",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


_MIN_EVAL_PTS = 10  # data points needed before warnings are meaningful

def _render_warning_box(
    critical: bool,
    early_warning: bool,
    flagged_vars: list[str],
    z_scores: dict[str, float],
    n_pts: int = 0,
) -> None:
    if n_pts < _MIN_EVAL_PTS:
        st.markdown(
            '<div class="gbd-box gbd-box-wait" style="font-size:0.85rem;display:flex;align-items:center;">'
            '⏳ Warte auf mehr Datenpunkte…</div>',
            unsafe_allow_html=True,
        )
        return

    if critical or early_warning:
        flagged = flagged_vars
        top_z = sorted(
            [(v, z) for v, z in z_scores.items() if abs(float(z)) > 2.0],
            key=lambda kv: abs(float(kv[1])), reverse=True,
        )[:3]

        def _sev_label(z: float) -> str:
            arrow = "↑" if z > 0 else "↓"
            level = "sehr hoch" if abs(z) >= 4 else "hoch"
            return f"Abweichung {arrow} {level}"

        flagged_readable = ", ".join(_VAR_NAMES.get(v, v) for v in flagged) if flagged else "–"
        top_z_str = " &nbsp;·&nbsp; ".join(
            f"<b>{_VAR_NAMES.get(v, v)}</b>: {_sev_label(float(z))}" for v, z in top_z
        ) if top_z else "–"

        if critical:
            box_cls, lbl_cls, label = "gbd-box-crit", "gbd-label-crit", "🔴 Kritisch"
            subtitle = "Viele Messwerte außerhalb des Normalbereichs"
        else:
            box_cls, lbl_cls, label = "gbd-box-warn", "gbd-label-warn", "🟡 Frühwarnung"
            subtitle = "Mehrere Messwerte außerhalb des Normalbereichs"
        st.markdown(
            f'<div class="gbd-box {box_cls}">'
            f'<div class="{lbl_cls}">{label}<span class="gbd-subtitle">{subtitle}</span></div>'
            f'<div class="gbd-vars">{flagged_readable}</div>'
            f'<div class="gbd-topz">{top_z_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gbd-box gbd-box-ok">'
            '<span class="gbd-label-ok">🟢 OK — kein Early-Warning</span>'
            '</div>',
            unsafe_allow_html=True,
        )


def _add_phase_bands(fig: go.Figure, segs) -> None:
    for s in segs:
        fig.add_vrect(
            x0=float(s.t_start),
            x1=float(s.t_end),
            fillcolor=_phase_color(str(s.phase)),
            opacity=0.10,
            line_width=0,
            layer="below",
        )


_VAR_NAMES: dict[str, str] = {
    "temp_C": "Temperatur",
    "pH": "pH-Wert",
    "feed_A_Lph": "Zulauf A",
    "feed_B_Lph": "Zulauf B",
    "agitator_rpm": "Rührerdrehzahl",
    "pressure_bar": "Druck",
    "dissolved_O2": "Gelöster Sauerstoff",
    "conductivity": "Leitfähigkeit",
    "turbidity": "Trübung",
    "level_L": "Füllstand",
    "flow_rate": "Durchfluss",
    "viscosity": "Viskosität",
}

_VAR_UNITS: dict[str, str] = {
    "temp_C": "°C",
    "pH": "–",
    "feed_A_Lph": "L/h",
    "feed_B_Lph": "L/h",
    "agitator_rpm": "rpm",
    "pressure_bar": "bar",
    "dissolved_O2": "%",
    "conductivity": "mS/cm",
    "turbidity": "NTU",
    "level_L": "L",
    "flow_rate": "L/h",
    "viscosity": "mPas",
}


def _batch_status(info) -> tuple[str, str]:
    """Returns (emoji, label) for a batch info object."""
    if info is None:
        return "⚪", "Unbekannt"
    is_anom = getattr(info, "is_anomalous", None)
    q_pass = getattr(info, "quality_pass", None)
    if is_anom == 0 and q_pass:
        return "🟢", "Gut"
    if is_anom == 1:
        return "🔴", "Auffällig"
    return "🟡", "Ungeprüft"


with tab_ov:
        st.subheader("Batch-Übersicht")
        if not all_points or not batches_by_id:
            st.info("Keine Batch-Daten geladen.")
        else:
            if "overview_batch" not in st.session_state:
                st.session_state["overview_batch"] = sorted(batches_by_id.keys())[0]

            col_list, col_detail = st.columns([1, 4], gap="large")

            with col_list:
                st.markdown("**Batches**")
                search_q = st.text_input(
                    "Suche", placeholder="🔍 Batch-ID suchen …",
                    label_visibility="collapsed", key="batch_search",
                ) or ""
                all_bids = sorted(batches_by_id.keys())
                filtered_bids = [b for b in all_bids if search_q.lower() in b.lower()] if search_q else all_bids
                with st.container(height=600):
                    for bid in filtered_bids:
                        emoji, label = _batch_status(batches_by_id.get(bid))
                        is_sel = st.session_state.get("overview_batch") == bid
                        if st.button(
                            f"{emoji} {bid}",
                            key=f"obatch_{bid}",
                            type="primary" if is_sel else "secondary",
                            use_container_width=True,
                        ):
                            st.session_state["overview_batch"] = bid
                            st.rerun()

            with col_detail:
                sel = st.session_state.get("overview_batch")
                if not sel:
                    st.info("Bitte einen Batch auswählen.")
                else:
                    info = batches_by_id.get(sel)
                    emoji, label = _batch_status(info)
                    _card_cls = {"Gut": "gbd-status-ok", "Auffällig": "gbd-status-bad"}.get(label, "gbd-status-unk")
                    _lbl_cls  = {"Gut": "gbd-status-lbl-ok", "Auffällig": "gbd-status-lbl-bad"}.get(label, "gbd-status-lbl-unk")
                    st.markdown(
                        f'<div class="gbd-status-card {_card_cls}">'
                        f'<span style="font-size:2.8rem;line-height:1;">{emoji}</span>'
                        f'<div><div class="{_lbl_cls}">{label}</div>'
                        f'<div class="gbd-status-sub">{sel}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    batch_pts = _batch_points(sel)
                    if not batch_pts:
                        st.info(f"Keine Datenpunkte für {sel}.")
                    else:
                        try:
                            profile = _cached_golden_profile()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Golden Profile nicht verfügbar: {exc}")
                            profile = {"rows": [], "meta": {}}

                        # Build golden profile index: (phase, variable) -> list of rows
                        gp_idx: dict[tuple[str, str], list[dict]] = {}
                        for r in profile.get("rows") or []:
                            try:
                                gp_idx.setdefault((str(r["phase"]), str(r["variable"])), []).append(r)
                            except Exception:  # noqa: BLE001
                                continue

                        # Phases in order of appearance
                        _seen_ov: set[str] = set()
                        phases_ov = [
                            str(s.phase) for s in phase_segments(batch_pts)
                            if str(s.phase) and str(s.phase) not in _seen_ov
                            and not _seen_ov.add(str(s.phase))  # type: ignore[func-returns-value]
                        ]
                        variables_ov = sorted({v for p in batch_pts for v in p.values.keys()})
                        segs_ov = phase_segments(batch_pts)

                        chart_cols = st.columns(2)
                        for vi, var in enumerate(variables_ov):
                            unit = _VAR_UNITS.get(var, "–")
                            name = _VAR_NAMES.get(var, var)

                            t_x = [float(p.t_pct) for p in batch_pts]
                            y_batch = [float(p.values.get(var, float("nan"))) for p in batch_pts]
                            opt_x: list[float] = []
                            opt_mean: list[float] = []
                            opt_lower: list[float] = []
                            opt_upper: list[float] = []
                            pt_colors: list[str] = []

                            for p in batch_pts:
                                t_b = int((float(p.t_pct) // 5) * 5)
                                gp_rows = gp_idx.get((str(p.phase), var), [])
                                row = next(
                                    (r for r in gp_rows if int(r.get("t_pct_bucket", -1)) == t_b),
                                    gp_rows[0] if gp_rows else None,
                                )
                                if row and var in p.values:
                                    std_v = float(row["std"]) if float(row["std"]) > 0 else 1.0
                                    z = abs(float(p.values[var]) - float(row["mean"])) / std_v
                                    pt_colors.append(
                                        "#E45756" if z >= 2.5 else "#EECA3B" if z >= 1.2 else "#54A24B"
                                    )
                                    opt_x.append(float(p.t_pct))
                                    opt_mean.append(float(row["mean"]))
                                    opt_lower.append(float(row["lower"]))
                                    opt_upper.append(float(row["upper"]))
                                else:
                                    pt_colors.append("#AAAAAA")

                            _n_scored = [c for c in pt_colors if c != "#AAAAAA"]
                            _red_pct = _n_scored.count("#E45756") / len(_n_scored) if _n_scored else 0
                            _yel_pct = _n_scored.count("#EECA3B") / len(_n_scored) if _n_scored else 0
                            if _red_pct >= 0.05:
                                status_ov = "🔴"
                            elif _red_pct > 0 or _yel_pct >= 0.10:
                                status_ov = "🟡"
                            else:
                                status_ov = "🟢"

                            fig_v = go.Figure()
                            _add_phase_bands(fig_v, segs_ov)
                            if opt_x:
                                fig_v.add_trace(go.Scatter(
                                    x=opt_x, y=opt_upper, mode="lines",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                                ))
                                fig_v.add_trace(go.Scatter(
                                    x=opt_x, y=opt_lower, mode="lines",
                                    line=dict(width=0), fill="tonexty",
                                    fillcolor="rgba(76,120,168,0.12)",
                                    showlegend=False, hoverinfo="skip",
                                ))
                                fig_v.add_trace(go.Scatter(
                                    x=opt_x, y=opt_mean, mode="lines",
                                    line=dict(width=1.5, dash="dot", color="rgba(76,120,168,0.55)"),
                                    showlegend=False, hoverinfo="skip",
                                ))
                            fig_v.add_trace(go.Scatter(
                                x=t_x, y=y_batch,
                                mode="lines+markers",
                                line=dict(color="rgba(50,50,50,0.45)", width=1.5),
                                marker=dict(color=pt_colors, size=5),
                                showlegend=False,
                                hovertemplate=f"<b>%{{y:.3f}} {unit}</b><br>Zeitfortschritt: %{{x:.1f}} %<extra></extra>",
                            ))
                            fig_v.update_layout(
                                height=220,
                                margin=dict(l=10, r=10, t=28, b=30),
                                xaxis_title="Zeitfortschritt (%)",
                                yaxis_title=unit,
                                title=dict(text=f"{status_ov} {name}", font=dict(size=13), x=0),
                            )

                            with chart_cols[vi % 2]:
                                st.plotly_chart(fig_v, use_container_width=True)


with tab_gp:
        st.subheader("Golden Profile")
        if not all_points:
            st.info("Timeseries file not loaded.")
        else:
            st.markdown(
                """
    Ein gutes **Golden Profile** entsteht aus **“guten” historischen Batches** (normal + Qualitätsfreigabe).

    - **Repräsentativ**: nur Batches mit stabilem Prozessverlauf (hier: `is_anomalous == 0` und `quality_pass == True`)
    - **Robust**: als Zentrum nutzen wir den **Median** (weniger empfindlich gegenüber Ausreißern)
    - **Kontrollgrenzen**: Band = **Median ± 2σ** (gibt erwartete Schwankungsbreite je Phase & Fortschritt)

    Der Graph unten markiert **alle Messpunkte**, die tatsächlich als Datengrundlage in das Golden Profile eingehen.
    """
            )
            try:
                profile = _cached_golden_profile()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build golden profile: {exc}")
                profile = {"rows": [], "meta": {}}

            good_batch_ids = {bid for bid, info in batches_by_id.items() if info.is_anomalous == 0 and getattr(info, "quality_pass", False)}
            total_pts = len(all_points)
            good_pts = sum(1 for p in all_points if p.batch_id in good_batch_ids)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Gute Batches</div>'
                            f'<div class="gbd-metric-val">{len(good_batch_ids)}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Alle Datenpunkte</div>'
                            f'<div class="gbd-metric-val">{total_pts}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Im Golden Profile</div>'
                            f'<div class="gbd-metric-val">{good_pts}</div></div>',
                            unsafe_allow_html=True)

            # Influence plot: show all points, highlight those from good batches
            variables_all = sorted({v for p in all_points for v in p.values.keys()})
            if variables_all:
                default_idx_pts = variables_all.index("temp_C") if "temp_C" in variables_all else 0
                var_pts = st.selectbox("Variable (Einflussplot)", options=variables_all, index=default_idx_pts, key="influence_var")

                max_points = st.slider("Max. Datenpunkte (Performance)", min_value=2000, max_value=60000, value=15000, step=1000)

                @st.cache_data(show_spinner=False)
                def _influence_data(variable: str, max_pts: int) -> dict:
                    pts = [p for p in all_points if variable in p.values]
                    # Deterministic downsample: keep all "good" points if possible, sample the rest.
                    good = [p for p in pts if p.batch_id in good_batch_ids]
                    bad = [p for p in pts if p.batch_id not in good_batch_ids]
                    if len(good) >= max_pts:
                        pts_sel = good[:max_pts]
                    else:
                        remaining = max_pts - len(good)
                        pts_sel = good + bad[:remaining]

                    return {
                        "t_pct": [float(p.t_pct) for p in pts_sel],
                        variable: [float(p.values[variable]) for p in pts_sel],
                        "phase": [str(p.phase) for p in pts_sel],
                        "used_for_profile": [p.batch_id in good_batch_ids for p in pts_sel],
                        "batch_id": [p.batch_id for p in pts_sel],
                    }

                data = _influence_data(var_pts, int(max_points))
                if not data["t_pct"]:
                    st.info("Keine Datenpunkte für den Einflussplot.")
                else:
                    fig_inf = px.scatter(
                        data,
                        x="t_pct",
                        y=var_pts,
                        color="phase",
                        color_discrete_map=_PHASE_COLOR_MAP,
                        symbol="used_for_profile",
                        hover_data={"batch_id": True, "phase": True, "t_pct": ":.2f", "used_for_profile": True},
                        render_mode="webgl",
                    )
                    fig_inf.update_traces(marker=dict(size=5, opacity=0.65))
                    # Deduplicate legend: show only the phase name, hide duplicate entries
                    _seen: set[str] = set()
                    def _fix_legend_entry(trace):
                        phase_name = trace.name.split(",")[0].strip()
                        if phase_name in _seen:
                            trace.update(showlegend=False)
                        else:
                            _seen.add(phase_name)
                            trace.update(name=phase_name)
                    fig_inf.for_each_trace(_fix_legend_entry)
                    fig_inf.update_layout(
                        height=380,
                        margin=dict(l=20, r=20, t=10, b=20),
                        xaxis_title="Zeitfortschritt (%)",
                        yaxis_title="Messwert",
                        legend_title_text="Phase",
                    )
                    st.plotly_chart(fig_inf, use_container_width=True)



with tab_mon:
        import time as _time

        for _k, _v in [
            ("mon_running", False), ("mon_n", 1), ("mon_interval_s", 2.0),
            ("mon_selected_vars", []), ("mon_use_dtw", True),
            ("mon_auto_vars", True), ("mon_auto_var_history", set()), ("mon_last_batch", ""),
            ("mon_consec_flags", {}),
            ("mon_show_bands", True), ("mon_distinct_colors", False),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        st.subheader("Live Monitoring")

        if not all_points:
            st.info("Zeitreihendaten nicht geladen.")
        else:
            batch_id = st.selectbox(
                "Batch",
                options=sorted({p.batch_id for p in all_points}),
                index=0,
                key="mon_batch",
            )
            # Reset auto-var history when a different batch is selected
            if st.session_state.mon_last_batch != batch_id:
                st.session_state.mon_auto_var_history = set()
                st.session_state.mon_consec_flags = {}
                st.session_state.mon_last_batch = batch_id

            pts = _batch_points(batch_id)

            if not pts:
                st.info(f"Keine Datenpunkte für {batch_id}.")
            else:
                variables_all = sorted({v for p in pts for v in p.values.keys()})
                if not st.session_state.mon_selected_vars:
                    _dv = "temp_C" if "temp_C" in variables_all else (variables_all[0] if variables_all else "")
                    st.session_state.mon_selected_vars = [_dv] if _dv else []
                selected_vars = (
                    [v for v in st.session_state.mon_selected_vars if v in variables_all]
                    or [variables_all[0]]
                )

                # Advance counter before rendering so slider reflects updated position
                if st.session_state.mon_running:
                    if st.session_state.mon_n < len(pts):
                        st.session_state.mon_n += 1
                    else:
                        st.session_state.mon_running = False

                # ── Control bar ───────────────────────────────────────────────────
                c_play, c_stop, c_slider, c_ai, c_set = st.columns([1, 1, 5, 3, 2], gap="small")

                with c_play:
                    play_icon = "⏸" if st.session_state.mon_running else "▶"
                    if st.button(play_icon, key="mon_play_btn", type="primary", use_container_width=True):
                        if st.session_state.mon_n >= len(pts):
                            st.session_state.mon_n = 1
                        st.session_state.mon_running = not st.session_state.mon_running
                        st.rerun()

                with c_stop:
                    if st.button("↺ Reset", key="mon_stop_btn", use_container_width=True):
                        st.session_state.mon_running = False
                        st.session_state.mon_n = 1
                        st.session_state.mon_auto_var_history = set()
                        st.session_state.mon_consec_flags = {}
                        st.rerun()

                with c_slider:
                    # Force slider to track mon_n during playback
                    if st.session_state.mon_running:
                        st.session_state["mon_pos_slider"] = max(1, min(st.session_state.mon_n, len(pts)))
                    slider_n = st.slider(
                        "Position",
                        min_value=1,
                        max_value=len(pts),
                        value=max(1, min(st.session_state.mon_n, len(pts))),
                        label_visibility="collapsed",
                        key="mon_pos_slider",
                    )
                    if not st.session_state.mon_running:
                        st.session_state.mon_n = slider_n

                with c_ai:
                    trigger_ai = st.button("🤖 KI Agenten fragen", key="mon_ai_btn", use_container_width=True)

                with c_set:
                    with st.popover("⚙ Einstellungen", use_container_width=True):
                        st.session_state.mon_auto_vars = st.checkbox(
                            "Auffällige Variablen automatisch einblenden",
                            value=st.session_state.mon_auto_vars,
                            key="settings_auto_vars",
                        )
                        _vars_disabled = st.session_state.mon_auto_vars
                        st.markdown(
                            "**Anzuzeigende Variablen**"
                            + (" *(automatisch)*" if _vars_disabled else "")
                        )
                        new_vars = st.multiselect(
                            "vars",
                            options=variables_all,
                            default=st.session_state.mon_selected_vars,
                            label_visibility="collapsed",
                            key="settings_vars",
                            disabled=_vars_disabled,
                        )
                        if new_vars and not _vars_disabled:
                            st.session_state.mon_selected_vars = new_vars
                        st.divider()
                        st.markdown("**Geschwindigkeit** (Sekunden / Datenpunkt)")
                        st.session_state.mon_interval_s = float(st.slider(
                            "speed",
                            min_value=0.2, max_value=5.0,
                            value=float(st.session_state.mon_interval_s),
                            step=0.2,
                            label_visibility="collapsed",
                            key="settings_speed",
                        ))
                        st.divider()
                        st.markdown("**Darstellung**")
                        st.session_state.mon_show_bands = st.checkbox(
                            "Optimalbänder anzeigen",
                            value=st.session_state.mon_show_bands,
                            key="settings_show_bands",
                        )
                        st.session_state.mon_distinct_colors = st.checkbox(
                            "Farbe nach Variable (statt nach Status)",
                            value=st.session_state.mon_distinct_colors,
                            key="settings_distinct_colors",
                        )
                        st.session_state.mon_use_dtw = st.checkbox(
                            "DTW-Ausrichtung verwenden (tslearn)",
                            value=st.session_state.mon_use_dtw,
                            key="settings_dtw",
                        )

                pts_n = pts[:max(1, st.session_state.mon_n)]

                # ── Profile & evals ───────────────────────────────────────────────
                try:
                    profile = _cached_golden_profile()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not build golden profile: {exc}")
                    profile = {"rows": [], "meta": {}}

                evals = (
                    evaluate_stream_dtw(pts_n, golden_profile=profile)
                    if st.session_state.mon_use_dtw
                    else evaluate_stream(
                        pts_n, golden_profile=profile,
                        window_size=5, t_pct_step=5, z_threshold=2.0,
                        early_warning_min_vars=2, critical_phase_ratio=0.60,
                    )
                )
                last = evals[-1]

                # ── Golden-profile index (used throughout: auto-vars, chart, table) ─
                gp_index: dict[tuple[str, int, str], dict] = {}
                for _r in profile.get("rows") or []:
                    try:
                        gp_index[(str(_r["phase"]), int(_r["t_pct_bucket"]), str(_r["variable"]))] = _r
                    except Exception:  # noqa: BLE001
                        continue

                def _t_bucket(t: float) -> int:
                    return int((float(t) // 5) * 5)

                def _band_flags_zscores(p) -> tuple[dict[str, bool], dict[str, float]]:
                    """Direct band comparison — consistent with displayed Golden Band."""
                    t_b = _t_bucket(p.t_pct)
                    flags: dict[str, bool] = {}
                    z_sc: dict[str, float] = {}
                    for var, x in p.values.items():
                        row = gp_index.get((str(p.phase), t_b, var))
                        if not row or float(row["std"]) <= 0:
                            continue
                        z = (float(x) - float(row["mean"])) / float(row["std"])
                        z_sc[var] = z
                        flags[var] = abs(z) > 2.0
                    return flags, z_sc

                # ── Auto-var: use direct band flags (consistent with graph/table) ──
                if st.session_state.mon_auto_vars and len(pts_n) >= _MIN_EVAL_PTS:
                    _cur_flags, _ = _band_flags_zscores(pts_n[-1])
                    _flagged_now = {v for v, outside in _cur_flags.items() if outside}
                    _consec = st.session_state.mon_consec_flags
                    for v in variables_all:
                        _consec[v] = (_consec.get(v, 0) + 1) if v in _flagged_now else 0
                    for v, cnt in _consec.items():
                        if cnt >= 1:
                            st.session_state.mon_auto_var_history.add(v)
                if st.session_state.mon_auto_vars:
                    _auto = [v for v in st.session_state.mon_auto_var_history if v in variables_all]
                    selected_vars = list(dict.fromkeys(selected_vars + _auto))

                # ── KI call — runs before chart so result is ready for right panel ─
                if trigger_ai:
                    with st.spinner("KI Agent analysiert den aktuellen Zeitpunkt…"):
                        try:
                            snap_text = generate_snapshot_report(
                                batch_id=batch_id,
                                phase=last.phase,
                                t_pct=last.t_pct,
                                z_scores=last.z_scores,
                                flags=last.flags,
                            )
                            st.session_state["_snap_report"] = snap_text
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Fehler: {exc}")

                # ── Metrics ───────────────────────────────────────────────────────
                mc1, mc2, mc3 = st.columns(3)
                _cur_direct_flags, _ = _band_flags_zscores(pts_n[-1])
                _flags_content = str(sum(1 for v in _cur_direct_flags.values() if v)) if len(pts_n) >= _MIN_EVAL_PTS else "–"
                with mc1:
                    st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Phase</div>'
                                f'<div class="gbd-metric-val">{last.phase}</div></div>',
                                unsafe_allow_html=True)
                with mc2:
                    st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Zeitfortschritt</div>'
                                f'<div class="gbd-metric-val">{last.t_pct:.2f} %</div></div>',
                                unsafe_allow_html=True)
                with mc3:
                    st.markdown(f'<div class="gbd-metric"><div class="gbd-metric-title">Auffällige Variablen</div>'
                                f'<div class="gbd-metric-val">{_flags_content}</div></div>',
                                unsafe_allow_html=True)

                # ── Build chart ───────────────────────────────────────────────────
                x = [p.t_pct for p in pts_n]
                _pt_direct = [_band_flags_zscores(p) for p in pts_n]
                flagged_pts = [any(f.values()) for f, _ in _pt_direct]
                colors = ["#E45756" if f else "#4C78A8" for f in flagged_pts]

                def _fmt_point_hover(e, var: str, val: float,
                                     direct_flags: dict[str, bool],
                                     direct_z: dict[str, float]) -> str:
                    flagged_vars = sorted([k for k, v in direct_flags.items() if v])
                    # Only show direction for variables that are actually outside the band
                    flagged_z = sorted(
                        [(v, direct_z[v]) for v in flagged_vars if v in direct_z],
                        key=lambda kv: abs(float(kv[1])), reverse=True,
                    )

                    def _sev(z: float) -> str:
                        direction = "↑" if z > 0 else "↓"
                        return f"{direction} {'sehr stark' if abs(z) >= 4 else 'stark' if abs(z) >= 2.5 else 'hoch'}"

                    lines = [f"<b>Phase: {e.phase} &nbsp;|&nbsp; {e.t_pct:.1f} %</b>", f"{var}: {val:.3f}"]
                    if flagged_vars:
                        lines.append("<br>⚠ <b>Außerhalb Normalbereich:</b>")
                        lines += [
                            f"&nbsp;&nbsp;• {_VAR_NAMES.get(v, v)}: {_sev(float(z))}"
                            for v, z in flagged_z
                        ]
                    return "<br>".join(lines)

                fig = go.Figure()
                segs = phase_segments(pts_n)
                _add_phase_bands(fig, segs)

                _seen_ph: set[str] = set()
                phases_present = [
                    str(s.phase) for s in segs
                    if str(s.phase) and str(s.phase) not in _seen_ph
                    and not _seen_ph.add(str(s.phase))  # type: ignore[func-returns-value]
                ]
                for ph in phases_present:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode="markers",
                        marker=dict(size=10, color=_phase_color(ph)),
                        name=f"Phase: {ph}", showlegend=True, hoverinfo="skip",
                    ))

                _Y2_VARS = {"agitator_rpm"}  # variables shown on the right axis
                _var_color_map = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(variables_all)}

                for var in selected_vars:
                    y = [float(p.values.get(var, float("nan"))) for p in pts_n]
                    hover_texts = [
                        _fmt_point_hover(evals[i], var, v, _pt_direct[i][0], _pt_direct[i][1])
                        for i, v in enumerate(y)
                    ]
                    _yax = "y2" if var in _Y2_VARS else "y"
                    # Per-variable flagging: dot is red only when THIS variable is outside its band
                    _var_outside = [_pt_direct[i][0].get(var, False) for i in range(len(pts_n))]
                    if st.session_state.mon_distinct_colors:
                        _line_c = _var_color_map.get(var, "#4C78A8")
                        _marker_colors = _line_c
                    else:
                        _line_c = "rgba(50,50,50,0.45)"
                        _marker_colors = ["#E45756" if f else "#4C78A8" for f in _var_outside]
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode="lines+markers",
                        marker=dict(color=_marker_colors, size=7),
                        line=dict(color=_line_c, width=2),
                        name=_VAR_NAMES.get(var, var), text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                        yaxis=_yax,
                    ))
                    # Flagged-point overlay (×) when using distinct colors
                    if st.session_state.mon_distinct_colors:
                        _fx = [x[i] for i, f in enumerate(_var_outside) if f]
                        _fy = [y[i] for i, f in enumerate(_var_outside) if f]
                        if _fx:
                            fig.add_trace(go.Scatter(
                                x=_fx, y=_fy, mode="markers",
                                marker=dict(symbol="x", color="#E45756", size=9, line=dict(width=2)),
                                name=f"⚠ {_VAR_NAMES.get(var, var)}", showlegend=False,
                                hoverinfo="skip", yaxis=_yax,
                            ))

                if st.session_state.mon_show_bands:
                    for var in selected_vars:
                        opt_upper_v: list[float] = []
                        opt_lower_v: list[float] = []
                        opt_mean_v: list[float] = []
                        opt_x_v: list[float] = []
                        for p in pts_n:
                            row = gp_index.get((str(p.phase), _t_bucket(p.t_pct), str(var)))
                            if not row:
                                continue
                            opt_x_v.append(float(p.t_pct))
                            opt_mean_v.append(float(row["mean"]))
                            opt_lower_v.append(float(row["lower"]))
                            opt_upper_v.append(float(row["upper"]))
                        if not opt_x_v:
                            continue
                        _yax = "y2" if var in _Y2_VARS else "y"
                        _band_color = (
                            _var_color_map.get(var, "#4C78A8").lstrip("#")
                            if st.session_state.mon_distinct_colors else "4C78A8"
                        )
                        _r, _g, _b = int(_band_color[0:2], 16), int(_band_color[2:4], 16), int(_band_color[4:6], 16)
                        fig.add_trace(go.Scatter(
                            x=opt_x_v, y=opt_upper_v, mode="lines",
                            line=dict(width=0), showlegend=False, hoverinfo="skip", yaxis=_yax,
                        ))
                        fig.add_trace(go.Scatter(
                            x=opt_x_v, y=opt_lower_v, mode="lines",
                            line=dict(width=0), fill="tonexty",
                            fillcolor=f"rgba({_r},{_g},{_b},0.10)",
                            showlegend=False, hoverinfo="skip", yaxis=_yax,
                        ))
                        fig.add_trace(go.Scatter(
                            x=opt_x_v, y=opt_mean_v, mode="lines",
                            line=dict(width=1.5, dash="dot", color=f"rgba({_r},{_g},{_b},0.5)"),
                            showlegend=False, hoverinfo="skip", yaxis=_yax,
                        ))

                _has_y2 = any(v in _Y2_VARS for v in selected_vars)
                fig.update_layout(
                    height=480, margin=dict(l=20, r=20, t=20, b=100),
                    xaxis_title="Zeitfortschritt (%)",
                    yaxis=dict(title="Messwert"),
                    yaxis2=dict(
                        title="rpm" if _has_y2 else "",
                        overlaying="y", side="right",
                        showgrid=False, visible=_has_y2,
                    ),
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.18,
                        xanchor="left", x=0,
                    ),
                )
                st.session_state["_monitor_fig"] = fig
                st.session_state["_monitor_last"] = last

                # ── Chart (left) + Analysis panel (right) ─────────────────────────
                st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
                col_chart, col_panel = st.columns([3, 2], gap="large")

                with col_chart:
                    st.plotly_chart(fig, use_container_width=True)

                with col_panel:
                    # Use current-point flags only — consistent with the table
                    _cur_flags_warn, _cur_z = _pt_direct[-1]
                    _cur_flagged_vars = sorted(v for v, out in _cur_flags_warn.items() if out)
                    _direct_ew = len(_cur_flagged_vars) >= 1
                    # Critical: proportion of current-phase points with any flag > 0.6
                    _cur_phase = pts_n[-1].phase
                    _phase_direct = [(p, _pt_direct[i]) for i, p in enumerate(pts_n) if p.phase == _cur_phase]
                    _phase_flagged = sum(1 for _, (_df, _) in _phase_direct if any(_df.values()))
                    _direct_critical = (_phase_flagged / len(_phase_direct)) > 0.6 if _phase_direct else False
                    _render_warning_box(
                        critical=_direct_critical,
                        early_warning=_direct_ew,
                        flagged_vars=_cur_flagged_vars,
                        z_scores=_cur_z,
                        n_pts=len(pts_n),
                    )
                    st.markdown("---")
                    st.markdown(
                        '<div class="gbd-ai-header">'
                        '<span style="font-size:1.3rem;">🤖</span>'
                        '<span class="gbd-ai-title">KI-Analyse</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    if st.session_state.get("_snap_report"):
                        with st.container(border=True):
                            st.markdown(st.session_state["_snap_report"])
                    else:
                        st.markdown(
                            '<div class="gbd-ai-ph">'
                            '<div style="font-size:2rem;margin-bottom:8px;">🤖</div>'
                            '<div style="font-weight:600;margin-bottom:4px;">Noch keine Analyse</div>'
                            '<div style="font-size:0.85rem;">Klicke auf <b>KI Agenten fragen</b>, um den '
                            'aktuellen Zeitpunkt analysieren zu lassen.</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                # ── Detail table ──────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### Detailansicht — Messwerte im Vergleich")
                last_pt = pts_n[-1]
                detail_rows = []
                for var in variables_all:
                    unit = _VAR_UNITS.get(var, "–")
                    cur_val = last_pt.values.get(var)
                    if cur_val is None:
                        continue
                    t_b = _t_bucket(last_pt.t_pct)
                    gp_row = gp_index.get((str(last_pt.phase), t_b, var))
                    golden_med: float | str = round(float(gp_row["mean"]), 3) if gp_row else "–"
                    golden_band: str = (
                        f"{float(gp_row['lower']):.3f} – {float(gp_row['upper']):.3f}" if gp_row else "–"
                    )
                    # Recompute z directly from the displayed band so status is always consistent
                    if gp_row and float(gp_row["std"]) > 0:
                        z_table = (float(cur_val) - float(gp_row["mean"])) / float(gp_row["std"])
                        abs_z = abs(z_table)
                    else:
                        z_table = None
                        abs_z = -1.0
                    outside = abs_z > 2.0
                    if outside:
                        status_icon, abw_label = "🔴", "hoch"
                    elif abs_z >= 1.6:
                        status_icon, abw_label = "🟡", "mittel"
                    elif z_table is not None:
                        status_icon, abw_label = "🟢", "–"
                    else:
                        status_icon, abw_label = "⚪", "–"
                    detail_rows.append({
                        "_abs_z": abs_z, "_flagged": outside,
                        "": status_icon,
                        "Messwert": _VAR_NAMES.get(var, var),
                        "Kennung": var, "Einheit": unit,
                        "Aktueller Wert": round(float(cur_val), 3),
                        "Golden Median": golden_med, "Golden Band": golden_band,
                        "Abweichung": abw_label,
                    })
                detail_rows.sort(key=lambda r: (-int(r["_flagged"]), -r["_abs_z"]))
                display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in detail_rows]
                if display_rows:
                    ui.table(pd.DataFrame(display_rows), key="detail_table")

                # ── Auto-advance ──────────────────────────────────────────────────
                if st.session_state.mon_running and st.session_state.mon_n < len(pts):
                    _time.sleep(float(st.session_state.mon_interval_s))
                    st.rerun()


with tab_rep:
        st.subheader("Report")
        batch_default = "A_B003"
        batch_id = st.text_input("Batch-ID", value=batch_default)

        dev_fig = _deviation_chart(batch_id)
        if dev_fig:
            st.markdown("**Abweichungen im Batchverlauf**")
            st.plotly_chart(dev_fig, use_container_width=True)

        prompt = f"Produktionsbericht für Batch {batch_id}"
        if st.button("Bericht erstellen"):
            with st.spinner("Agent läuft..."):
                try:
                    result = run_agent(prompt)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Agent fehlgeschlagen: {exc}")
                    result = {}

            analysis = result.get("batch_analysis", {}) if isinstance(result, dict) else {}
            report = result.get("report", "") if isinstance(result, dict) else ""
            all_driver_vars = [d.get("variable") for d in (analysis.get("top_drivers") or [])[:5] if d.get("variable")]
            critical_phase = analysis.get("critical_phase") or ""

            if report:
                html = report
                if critical_phase:
                    html = html.replace(
                        critical_phase,
                        f"<span style='font-weight:700; background-color:rgba(245,158,11,0.15); padding:1px 5px; border-radius:4px'>{critical_phase}</span>",
                    )
                for v in all_driver_vars:
                    html = html.replace(str(v), f"<strong>{v}</strong>")
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("Kein Bericht erhalten.")

        with st.expander("Planerausgabe / vollständige Antwort"):
            if st.button("Agent ausführen (Plan/Antwort anzeigen)"):
                with st.spinner("Agent läuft..."):
                    result = run_agent(prompt)
                st.write(result.get("plan", ""))
                st.write(result.get("response", ""))

        if "history" not in st.session_state:
            st.session_state.history = []

        with st.form("prompt_form"):
            user_prompt = st.text_area(
                "Frage den Assistenten",
                placeholder="z.B.: Erkläre mir, was in Batch A_B152 schiefgelaufen ist, sodass die Qualität nicht eingehalten werden konnte.",
                height=100,
            )
            submitted = st.form_submit_button("Agent starten")

        if submitted:
            prompt = user_prompt.strip()
            if not prompt:
                st.warning("Bitte gib eine Anfrage ein.")
            else:
                with st.spinner("Agent läuft..."):
                    try:
                        result = run_agent(prompt)
                        st.session_state.history.append(
                            {
                                "prompt": prompt,
                                "plan": result["plan"],
                                "response": result["response"],
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Agent fehlgeschlagen: {exc}")

        if st.session_state.history:
            st.subheader("Letzte Ausführungen")
            for idx, item in enumerate(reversed(st.session_state.history), start=1):
                st.markdown(f"### Durchlauf {idx}")
                st.markdown(f"**Anfrage**: {item['prompt']}")
                with st.expander("Planerausgabe"):
                    st.write(item["plan"])
                st.markdown("**Antwort**")
                st.write(item["response"])
