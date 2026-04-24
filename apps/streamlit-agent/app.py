from __future__ import annotations

from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from agent.batches import load_case_a_batches, resolve_case_a_batches_path
from agent.graph import run_agent
from agent.golden_profile import build_case_a_golden_profile
from agent.monitoring import evaluate_stream, evaluate_stream_dtw
from agent.timeseries import load_case_a_timeseries, phase_segments, resolve_case_a_timeseries_path
from agent.driver_analysis import analyze_batch_against_golden_profile
from agent.multivariate import score_isolation_forest


st.set_page_config(page_title="Agentic Workshop Quickstarter", page_icon="🤖", layout="wide")


def _check_password() -> bool:
    if st.session_state.get("_authenticated"):
        return True

    expected = st.secrets.get("auth", {}).get("password", "")
    if not expected:
        return True  # No password configured → open access (local dev)

    st.title("Agentic Workshop Quickstarter")
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

st.title("Agentic Workshop Quickstarter")

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

tabs = st.tabs(["Golden Profile", "Live Monitoring", "Report"])


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


def _render_warning_box(last) -> None:
    if last.critical or last.early_warning:
        flagged_in_window = last.flagged_variables_window
        top_z = sorted(last.z_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
        flagged_lines = "\n".join(f"- **`{v}`**" for v in flagged_in_window) if flagged_in_window else "- –"
        top_z_lines = "\n".join(f"- **{v}**: {float(z):+.2f}σ" for v, z in top_z) if top_z else "- –"
        if last.critical:
            st.error(
                f"**CRITICAL** — >60 % Überschreitungen in der aktuellen Phase\n\n"
                f"**Außerhalb des optimalen Fensters:**\n{flagged_lines}\n\n"
                f"**Größte Abweichungen (|z|):**\n{top_z_lines}"
            )
        else:
            st.warning(
                f"**EARLY WARNING** — ≥2 Variablen im Sliding Window außerhalb\n\n"
                f"**Außerhalb des optimalen Fensters:**\n{flagged_lines}\n\n"
                f"**Größte Abweichungen (|z|):**\n{top_z_lines}"
            )
    else:
        st.success("OK: noch kein Early-Warning.")


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


with tabs[0]:
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
        c1.metric("Gute Batches", str(len(good_batch_ids)))
        c2.metric("Alle Datenpunkte", str(total_pts))
        c3.metric("Datenpunkte im Golden Profile", str(good_pts))

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



with tabs[1]:
    if st.session_state.get("_monitor_fullscreen"):
        fig_fs = st.session_state.get("_monitor_fig")
        last_fs = st.session_state.get("_monitor_last")
        if st.button("✕ Vollbild beenden"):
            st.session_state["_monitor_fullscreen"] = False
            st.rerun()
        if fig_fs is not None and last_fs is not None:
            fig_fs = go.Figure(fig_fs)
            fig_fs.update_layout(height=700)
            st.plotly_chart(fig_fs, use_container_width=True)
            _render_warning_box(last_fs)
        else:
            st.info("Noch kein Batch geladen. Bitte zuerst im normalen Modus einen Batch auswählen.")
    else:
        st.subheader("Live Monitoring")
    if not all_points:
        st.info("Zeitreihendaten nicht geladen.")
    elif not st.session_state.get("_monitor_fullscreen"):
        batch_id = st.selectbox("Batch", options=sorted({p.batch_id for p in all_points}) or ["A_B003"], index=0)
        pts = _batch_points(batch_id)
        if not pts:
            st.info(f"Keine Datenpunkte für {batch_id}.")
        else:
            mode = st.radio("Modus", options=["Schieberegler", "Simuliertes Streaming"], horizontal=True)
            use_dtw = st.checkbox("DTW-Ausrichtung verwenden (tslearn)", value=True)
            variables_all = sorted({v for p in pts for v in p.values.keys()})
            default_var = "temp_C" if "temp_C" in variables_all else (variables_all[0] if variables_all else "temp_C")
            selected_vars = st.multiselect(
                "Anzuzeigende Variablen",
                options=variables_all,
                default=[default_var] if default_var in variables_all else [],
            )
            if not selected_vars:
                selected_vars = [default_var]

            if mode == "Schieberegler":
                n = st.slider("Simulierte Datenpunkte", min_value=1, max_value=len(pts), value=min(30, len(pts)))
                pts_n = pts[:n]
            else:
                import time

                if "stream_running" not in st.session_state:
                    st.session_state.stream_running = False
                if "stream_paused" not in st.session_state:
                    st.session_state.stream_paused = False
                if "stream_n" not in st.session_state:
                    st.session_state.stream_n = 0
                if "stream_status_msg" not in st.session_state:
                    st.session_state.stream_status_msg = ""
                if "stream_interval_s" not in st.session_state:
                    st.session_state.stream_interval_s = 2.0

                st.session_state.stream_interval_s = float(
                    st.slider(
                        "Geschwindigkeit (Sekunden pro Datenpunkt)",
                        min_value=0.2,
                        max_value=5.0,
                        value=float(st.session_state.stream_interval_s),
                        step=0.2,
                    )
                )

                c_start, c_pause, c_stop = st.columns(3)
                if c_start.button("Start / Fortsetzen", disabled=st.session_state.stream_running and not st.session_state.stream_paused):
                    st.session_state.stream_running = True
                    st.session_state.stream_paused = False

                if c_pause.button("Pause", disabled=not st.session_state.stream_running or st.session_state.stream_paused):
                    st.session_state.stream_paused = True
                    st.session_state.stream_running = False
                    st.rerun()

                if c_stop.button("Stop", disabled=not st.session_state.stream_running and not st.session_state.stream_paused):
                    st.session_state.stream_running = False
                    st.session_state.stream_paused = False

                # Advance one point per rerun (non-blocking).
                # We increment first, render the UI, then schedule the next rerun at the bottom.
                if st.session_state.stream_running and st.session_state.stream_n < len(pts):
                    st.session_state.stream_n += 1

                # Freeze view at current n (also during pause)
                pts_n = pts[: max(1, st.session_state.stream_n)]

            try:
                profile = _cached_golden_profile()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build golden profile: {exc}")
                profile = {"rows": [], "meta": {}}

            evals = (
                evaluate_stream_dtw(pts_n, golden_profile=profile)
                if use_dtw
                else evaluate_stream(
                    pts_n,
                    golden_profile=profile,
                    window_size=5,
                    t_pct_step=5,
                    z_threshold=2.0,
                    early_warning_min_vars=2,
                    critical_phase_ratio=0.60,
                )
            )
            last = evals[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("Phase", last.phase)
            c2.metric("Zeitfortschritt (%)", f"{last.t_pct:.2f}")
            c3.metric("Auffällige Variablen", str(len(last.flagged_variables_window)))

            # Plot: selected variables vs t_pct with flagged points highlighted + phase bands
            x = [p.t_pct for p in pts_n]
            flagged = [any(e.flags.values()) for e in evals]
            colors = ["#E45756" if f else "#4C78A8" for f in flagged]

            def _fmt_point_hover(e, var: str, val: float) -> str:
                flagged_vars = sorted([k for k, v in e.flags.items() if v])
                top_z = sorted(e.z_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:4]

                def _sev(z: float) -> str:
                    a = abs(z)
                    direction = "↑" if z > 0 else "↓"
                    label = "stark" if a >= 2.5 else "mittel" if a >= 1.2 else "gering"
                    return f"{direction} {label}"

                lines = [
                    f"<b>Phase: {e.phase} &nbsp;|&nbsp; {e.t_pct:.1f} %</b>",
                    f"{var}: {val:.3f}",
                ]
                if flagged_vars:
                    lines.append("<br>⚠ <b>Außerhalb Normalbereich:</b>")
                    lines += [f"&nbsp;&nbsp;• {v}" for v in flagged_vars]
                if top_z:
                    lines.append("<br><b>Stärkste Abweichungen:</b>")
                    lines += [f"&nbsp;&nbsp;• {v}: {_sev(float(z))}" for v, z in top_z]
                return "<br>".join(lines)

            fig = go.Figure()
            segs = phase_segments(pts_n)
            _add_phase_bands(fig, segs)

            # Phase legend in order of appearance (left to right in plot)
            _seen_ph: set[str] = set()
            phases_present = [str(s.phase) for s in segs if str(s.phase) and str(s.phase) not in _seen_ph and not _seen_ph.add(str(s.phase))]  # type: ignore[func-returns-value]
            for ph in phases_present:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=10, color=_phase_color(ph)),
                        name=f"Phase: {ph}",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

            # Build fast lookup for optimal ranges: (phase, t_bucket, var) -> row
            gp_index: dict[tuple[str, int, str], dict] = {}
            for r in profile.get("rows") or []:
                try:
                    gp_index[(str(r["phase"]), int(r["t_pct_bucket"]), str(r["variable"]))] = r
                except Exception:  # noqa: BLE001
                    continue

            def _t_bucket(t: float) -> int:
                return int((float(t) // 5) * 5)

            for var in selected_vars:
                y = [float(p.values.get(var, float("nan"))) for p in pts_n]
                hover_texts = [_fmt_point_hover(e, var, v) for e, v in zip(evals, y)]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        marker=dict(color=colors, size=7),
                        line=dict(width=2),
                        name=var,
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

            # Overlay control band for selected variables for the current phase (bucketed)
            for var in selected_vars:
                # Per-point optimal range lookup (updates automatically when phase changes mid-run)
                opt_mean: list[float] = []
                opt_lower: list[float] = []
                opt_upper: list[float] = []
                opt_x: list[float] = []
                for p in pts_n:
                    key = (str(p.phase), _t_bucket(p.t_pct), str(var))
                    row = gp_index.get(key)
                    if not row:
                        continue
                    opt_x.append(float(p.t_pct))
                    opt_mean.append(float(row["mean"]))
                    opt_lower.append(float(row["lower"]))
                    opt_upper.append(float(row["upper"]))

                if not opt_x:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=opt_x,
                        y=opt_upper,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=opt_x,
                        y=opt_lower,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(76,120,168,0.10)",
                        name=f"Optimalband ({var})",
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=opt_x,
                        y=opt_mean,
                        mode="lines",
                        line=dict(width=2, dash="dot"),
                        name=f"Optimaler Median ({var})",
                        hoverinfo="skip",
                    )
                )

            fig.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Zeitfortschritt (%)",
                yaxis_title="Messwert",
            )
            st.session_state["_monitor_fig"] = fig
            st.session_state["_monitor_last"] = last

            st.plotly_chart(fig, use_container_width=True)

            if st.button("⛶ Vollbild"):
                st.session_state["_monitor_fullscreen"] = True
                st.rerun()

            _render_warning_box(last)

            st.markdown("**Erklärbarkeit (welcher Messwert den Batch wie stark beeinflusst)**")
            if not evals:
                st.info("Noch keine Auswertung.")
            else:
                last_eval = evals[-1]
                if not last_eval.z_scores:
                    st.info("Noch keine Abweichungswerte für diesen Punkt verfügbar.")
                else:
                    contrib = [
                        {"variable": v, "z": float(z), "abs_z": abs(float(z)), "flagged": bool(last_eval.flags.get(v, False))}
                        for v, z in last_eval.z_scores.items()
                    ]
                    contrib.sort(key=lambda r: r["abs_z"], reverse=True)
                    top_k = st.slider("Anzahl anzuzeigender Variablen", min_value=3, max_value=min(20, len(contrib)), value=min(8, len(contrib)))
                    contrib_top = contrib[:top_k]

                    fig_bar = px.bar(
                        contrib_top,
                        x="abs_z",
                        y="variable",
                        orientation="h",
                        color="flagged",
                        color_discrete_map={True: "#E45756", False: "#4C78A8"},
                        hover_data={"z": ":.2f", "abs_z": ":.2f", "flagged": True},
                        title="Einfluss am aktuellen Messpunkt",
                    )
                    fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Abweichung", yaxis_title="")
                    st.plotly_chart(fig_bar, use_container_width=True)



            # Schedule next tick *after* rendering, so the live view updates.
            if mode == "Simuliertes Streaming" and st.session_state.get("stream_running") and st.session_state.get("stream_n", 0) < len(pts):
                import time

                time.sleep(float(st.session_state.get("stream_interval_s", 2.0)))
                st.rerun()


with tabs[2]:
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
                    f"<span style='font-weight:700; background-color:#FFF3CD; padding:1px 5px; border-radius:4px'>{critical_phase}</span>",
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
        placeholder="z.B.: Berechne (12 * 8) + 5 und erkläre kurz.",
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
