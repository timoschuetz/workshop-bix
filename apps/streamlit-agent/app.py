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


st.set_page_config(page_title="Agentic Workshop Quickstarter", page_icon="🤖")
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

    fig = go.Figure()
    segs = phase_segments(pts)
    _add_phase_bands(fig, segs)

    phases_present = sorted({str(s.phase) for s in segs if str(s.phase)})
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
        hovertemplate="t_pct=%{x:.1f}<br>max|z|=%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Zeitfortschritt (%)",
        yaxis_title="Stärke der Abweichung",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


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
        c1.metric("Good batches", str(len(good_batch_ids)))
        c2.metric("All points", str(total_pts))
        c3.metric("Points used for Golden Profile", str(good_pts))

        # Influence plot: show all points, highlight those from good batches
        variables_all = sorted({v for p in all_points for v in p.values.keys()})
        if variables_all:
            default_idx_pts = variables_all.index("temp_C") if "temp_C" in variables_all else 0
            var_pts = st.selectbox("Influence plot variable", options=variables_all, index=default_idx_pts, key="influence_var")

            max_points = st.slider("Max points to render (performance)", min_value=2000, max_value=60000, value=15000, step=1000)

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
                st.info("No points for influence plot.")
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
    st.subheader("Live Monitoring")
    if not all_points:
        st.info("Timeseries file not loaded.")
    else:
        batch_id = st.selectbox("Batch", options=sorted({p.batch_id for p in all_points}) or ["A_B003"], index=0)
        pts = _batch_points(batch_id)
        if not pts:
            st.info(f"No points for {batch_id}.")
        else:
            mode = st.radio("Mode", options=["Slider", "Simulated Streaming"], horizontal=True)
            use_dtw = st.checkbox("Use DTW alignment (tslearn)", value=False)
            variables_all = sorted({v for p in pts for v in p.values.keys()})
            default_var = "temp_C" if "temp_C" in variables_all else (variables_all[0] if variables_all else "temp_C")
            selected_vars = st.multiselect(
                "Variables to display",
                options=variables_all,
                default=[default_var] if default_var in variables_all else [],
            )
            if not selected_vars:
                selected_vars = [default_var]

            if mode == "Slider":
                n = st.slider("Simulated ingested points", min_value=1, max_value=len(pts), value=min(30, len(pts)))
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
                        "Streaming speed (seconds per point)",
                        min_value=0.2,
                        max_value=5.0,
                        value=float(st.session_state.stream_interval_s),
                        step=0.2,
                    )
                )

                c_start, c_pause, c_stop = st.columns(3)
                if c_start.button("Start / Resume", disabled=st.session_state.stream_running and not st.session_state.stream_paused):
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
            c2.metric("t_pct", f"{last.t_pct:.2f}")
            c3.metric("Flagged vars (window)", str(len(last.flagged_variables_window)))

            # Plot: selected variables vs t_pct with flagged points highlighted + phase bands
            x = [p.t_pct for p in pts_n]
            flagged = [any(e.flags.values()) for e in evals]
            colors = ["#E45756" if f else "#4C78A8" for f in flagged]

            def _fmt_point_hover(e):
                flagged_vars = [k for k, v in e.flags.items() if v]
                flagged_vars = sorted(flagged_vars)
                top_z = sorted(e.z_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:3]
                top_z_txt = ", ".join([f"{k}={float(z):+.2f}" for k, z in top_z]) if top_z else "-"
                flagged_txt = ", ".join(flagged_vars) if flagged_vars else "-"
                return f"phase={e.phase}<br>t_bucket={e.t_pct_bucket}<br>flagged={flagged_txt}<br>top|z|={top_z_txt}"

            hover_extra = [_fmt_point_hover(e) for e in evals]

            fig = go.Figure()
            segs = phase_segments(pts_n)
            _add_phase_bands(fig, segs)

            # Phase legend (dummy traces so vrect colors are explained)
            phases_present = sorted({str(s.phase) for s in segs if str(s.phase)})
            for ph in phases_present:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=10, color=_phase_color(ph)),
                        name=f"phase: {ph}",
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
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        marker=dict(color=colors, size=7),
                        line=dict(width=2),
                        name=var,
                        text=hover_extra,
                        hovertemplate=f"t_pct=%{{x:.2f}}<br>{var}=%{{y:.3f}}<br>%{{text}}<extra></extra>",
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
                        name=f"Golden band ({var})",
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=opt_x,
                        y=opt_mean,
                        mode="lines",
                        line=dict(width=2, dash="dot"),
                        name=f"Golden median ({var})",
                        hoverinfo="skip",
                    )
                )

            fig.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Zeitfortschritt (%)",
                yaxis_title="Messwert",
            )
            st.plotly_chart(fig, use_container_width=True)

            if last.critical or last.early_warning:
                flagged_in_window = last.flagged_variables_window
                top_z = sorted(last.z_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
                flagged_lines = "\n".join(f"- **`{v}`**" for v in flagged_in_window) if flagged_in_window else "- –"
                top_z_lines = "\n".join(
                    f"- **{v}**: {float(z):+.2f}σ" for v, z in top_z
                ) if top_z else "- –"
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

            st.markdown("**Explainability (which metric affects the batch, and by how much)**")
            if not evals:
                st.info("No evaluations yet.")
            else:
                last_eval = evals[-1]
                if not last_eval.z_scores:
                    st.info("No z-scores available yet for this point (missing golden profile rows / std==0).")
                else:
                    # Point-level contribution: absolute z-score per variable
                    contrib = [
                        {"variable": v, "z": float(z), "abs_z": abs(float(z)), "flagged": bool(last_eval.flags.get(v, False))}
                        for v, z in last_eval.z_scores.items()
                    ]
                    contrib.sort(key=lambda r: r["abs_z"], reverse=True)
                    top_k = st.slider("Top variables to show", min_value=3, max_value=min(20, len(contrib)), value=min(8, len(contrib)))
                    contrib_top = contrib[:top_k]

                    fig_bar = px.bar(
                        contrib_top,
                        x="abs_z",
                        y="variable",
                        orientation="h",
                        color="flagged",
                        color_discrete_map={True: "#E45756", False: "#4C78A8"},
                        hover_data={"z": ":.2f", "abs_z": ":.2f", "flagged": True},
                        title="Current point impact (|z| vs Golden Profile)",
                    )
                    fig_bar.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="|z|", yaxis_title="")
                    st.plotly_chart(fig_bar, use_container_width=True)


            recent = evals[-10:]
            st.dataframe(
                [
                    {
                        "idx": e.idx,
                        "phase": e.phase,
                        "t_pct": round(e.t_pct, 3),
                        "t_bucket": e.t_pct_bucket,
                        "flagged_vars": sorted([k for k, v in e.flags.items() if v]),
                        "early_warning": e.early_warning,
                        "critical": e.critical,
                    }
                    for e in recent
                ],
                use_container_width=True,
            )

            # Schedule next tick *after* rendering, so the live view updates.
            if mode == "Simulated Streaming" and st.session_state.get("stream_running") and st.session_state.get("stream_n", 0) < len(pts):
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
    if st.button("Generate report (Agent)"):
        with st.spinner("Running agent..."):
            try:
                result = run_agent(prompt)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent failed: {exc}")
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
            st.info("No report returned.")

    with st.expander("Planner output / full response"):
        if st.button("Run agent (show plan/response)"):
            with st.spinner("Running agent..."):
                result = run_agent(prompt)
            st.write(result.get("plan", ""))
            st.write(result.get("response", ""))

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("prompt_form"):
    user_prompt = st.text_area(
        "Ask the assistant",
        placeholder="Try: Calculate (12 * 8) + 5 and explain briefly.",
        height=100,
    )
    submitted = st.form_submit_button("Run agent")

if submitted:
    prompt = user_prompt.strip()
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Running agent..."):
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
                st.error(f"Agent failed: {exc}")

if st.session_state.history:
    st.subheader("Recent Runs")
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"### Run {idx}")
        st.markdown(f"**Prompt**: {item['prompt']}")
        with st.expander("Planner output"):
            st.write(item["plan"])
        st.markdown("**Response**")
        st.write(item["response"])
