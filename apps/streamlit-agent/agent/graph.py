from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .driver_analysis import analyze_batch_against_golden_profile, infer_batch_id
from .timeseries import load_case_a_timeseries, resolve_case_a_timeseries_path
from .tools import TOOLS
from .golden_profile import build_case_a_golden_profile


@dataclass
class OpenRouterSettings:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "agentic-workshop-quickstarter"
    app_url: str = "https://example.com"


def get_settings() -> OpenRouterSettings:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "agentic-workshop-quickstarter").strip()
    app_url = os.getenv("OPENROUTER_APP_URL", "https://example.com").strip()

    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in environment.")
    if not model:
        raise ValueError("Missing OPENROUTER_MODEL in environment.")

    return OpenRouterSettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        app_name=app_name,
        app_url=app_url,
    )


class AgentState(TypedDict):
    user_input: str
    plan: str
    response: str
    golden_profile: dict[str, Any]
    batch_analysis: dict[str, Any]
    report: str
    messages: Annotated[list, add_messages]


class AgentUpdate(TypedDict, total=False):
    user_input: str
    plan: str
    response: str
    golden_profile: dict[str, Any]
    batch_analysis: dict[str, Any]
    report: str
    messages: Annotated[list, add_messages]


def _build_llm(settings: OpenRouterSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        default_headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": settings.app_name,
        },
    )


def planner_node(state: AgentState) -> AgentUpdate:
    llm = _build_llm(get_settings())
    planner_prompt = (
        "You are a planning assistant. Produce a short 2-4 bullet plan for solving "
        "the user request. Keep it concrete."
    )
    plan_message = llm.invoke(
        [HumanMessage(content=f"{planner_prompt}\n\nUser request: {state['user_input']}")]
    )
    plan_text = str(plan_message.content)
    return {"plan": plan_text}


def golden_profile_node(state: AgentState) -> AgentUpdate:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        profile = build_case_a_golden_profile(repo_root, t_pct_step=5)
    except Exception as exc:  # noqa: BLE001
        profile = {"meta": {"case": "A", "error": str(exc)}, "rows": []}
    return {"golden_profile": profile}


def batch_analysis_node(state: AgentState) -> AgentUpdate:
    repo_root = Path(__file__).resolve().parents[3]
    batch_id = infer_batch_id(state.get("user_input", ""), default="A_B003")
    try:
        ts_path = resolve_case_a_timeseries_path(repo_root)
        all_points = load_case_a_timeseries(ts_path)
        points = [p for p in all_points if p.batch_id == batch_id]
        summary = analyze_batch_against_golden_profile(
            points,
            golden_profile=state.get("golden_profile", {"rows": [], "meta": {}}),
            window_size=5,
            t_pct_step=5,
            z_threshold=2.0,
        )
        analysis: dict[str, Any] = {
            "batch_id": batch_id,
            "anomaly_score": summary.anomaly_score,
            "top_drivers": summary.top_drivers[:10],
            "critical_phase": summary.critical_phase,
            "z_max": summary.z_max,
        }
    except Exception as exc:  # noqa: BLE001
        analysis = {"batch_id": batch_id, "error": str(exc), "anomaly_score": 0.0, "top_drivers": [], "critical_phase": None, "z_max": None}
    return {"batch_analysis": analysis}


_VARIABLE_NAMES: dict[str, str] = {
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


def _severity(mean_abs_z: float) -> tuple[str, str]:
    if mean_abs_z >= 2.5:
        return "hoch", "🔴"
    if mean_abs_z >= 1.2:
        return "mittel", "🟡"
    return "gering", "🟢"


def report_node(state: AgentState) -> AgentUpdate:
    llm = _build_llm(get_settings())
    analysis = state.get("batch_analysis", {}) or {}
    top_drivers = analysis.get("top_drivers", [])[:5]
    critical_phase = analysis.get("critical_phase") or "unbekannt"
    batch_id = analysis.get("batch_id", "unbekannt")

    table_rows = []
    for d in top_drivers:
        if not d.get("variable") or d.get("mean_abs_z") is None:
            continue
        label, ampel = _severity(float(d["mean_abs_z"]))
        var = d["variable"]
        name = _VARIABLE_NAMES.get(var, var)
        table_rows.append(f"| {name} | {var} | {ampel} {label} |")

    if table_rows:
        driver_table = (
            "| Name | Messwert | Abweichung |\n"
            "|---|---|---|\n"
            + "\n".join(table_rows)
        )
    else:
        driver_table = "Keine auffälligen Messwerte."

    system_prompt = (
        "Du bist ein Produktionsassistent, der Berichte für Maschinenführer in einer Fabrik schreibt. "
        "Deine Leser haben keine technische Ausbildung und kennen keine Fachbegriffe. "
        "Halte dich strikt an folgende Regeln:\n"
        "1. Schreibe ausschließlich auf Deutsch.\n"
        "2. Verwende KEINE technischen Begriffe wie Z-Wert, Anomalie-Score, Standardabweichung, Median, Golden Profile oder ähnliches.\n"
        "3. Nenne KEINE konkreten Zahlenwerte. Beschreibe Abweichungen nur mit 'gering', 'mittel' oder 'hoch'.\n"
        "4. Nenne NICHT, wie viele Messpunkte betroffen waren.\n"
        "5. Schreibe in kurzen, klaren Sätzen – so wie du es einem Kollegen auf dem Hallenflur erklären würdest.\n"
        "6. Die auffälligen Messwerte sind bereits als Tabelle mit Ampel vorgegeben. "
        "Übernimm diese Tabelle exakt so in den Bericht – ergänze sie nicht und verändere sie nicht.\n"
        "7. Verwende für die drei Abschnitte die folgenden Markdown-Überschriften exakt so: "
        "'### Was ist passiert?', '### Auffällige Messwerte', '### Was ist zu tun?'. "
        "Keine nummerierten Listen.\n"
        "8. Handlungsempfehlungen müssen konkrete Anweisungen sein, was der Maschinenführer beim nächsten Batch tun soll."
    )

    user_prompt = (
        f"Erstelle einen Produktionsbericht für Batch {batch_id}.\n\n"
        f"Kritischste Phase: {critical_phase}\n\n"
        f"Auffällige Messwerte (bereits als Tabelle – exakt so übernehmen):\n\n{driver_table}\n\n"
        "Strukturiere den Bericht mit diesen drei Abschnitten:\n\n"
        "### Was ist passiert?\n"
        "(Kurze Zusammenfassung in 1-2 Sätzen. Nenne die kritische Phase namentlich.)\n\n"
        "### Auffällige Messwerte\n"
        "(Tabelle hier exakt einfügen.)\n\n"
        "### Was ist zu tun?\n"
        "(Eine konkrete Anweisung pro auffälligem Messwert für den nächsten Batch.)"
    )

    try:
        msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        report = str(msg.content).strip()
    except Exception as exc:  # noqa: BLE001
        report = f"Report generation failed: {exc}"

    return {"report": report}


def actor_node(state: AgentState) -> AgentUpdate:
    llm_with_tools = _build_llm(get_settings()).bind_tools(TOOLS)
    system = (
        "You are an assistant that can call tools when needed. "
        "Use tools for arithmetic. Keep the answer concise."
    )
    first_response: AIMessage = llm_with_tools.invoke(
        [
            HumanMessage(content=system),
            HumanMessage(content=f"Golden profile (serialised dict):\n{state.get('golden_profile', {})}"),
            HumanMessage(content=f"Batch analysis (serialised dict):\n{state.get('batch_analysis', {})}"),
            HumanMessage(content=f"Draft report:\n{state.get('report', '')}"),
            HumanMessage(content=f"Plan:\n{state['plan']}"),
            HumanMessage(content=f"User request:\n{state['user_input']}"),
        ]
    )

    tool_messages: list[ToolMessage] = []
    if first_response.tool_calls:
        tools_by_name = {tool.name: tool for tool in TOOLS}
        for call in first_response.tool_calls:
            tool_name = call["name"]
            tool = tools_by_name.get(tool_name)
            if tool is None:
                content = f"unknown_tool: {tool_name}"
            else:
                content = tool.invoke(call["args"])
            tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    if tool_messages:
        final_response = llm_with_tools.invoke(
            [
                HumanMessage(content=system),
                HumanMessage(content=f"Golden profile (serialised dict):\n{state.get('golden_profile', {})}"),
                HumanMessage(content=f"Batch analysis (serialised dict):\n{state.get('batch_analysis', {})}"),
                HumanMessage(content=f"Draft report:\n{state.get('report', '')}"),
                HumanMessage(content=f"Plan:\n{state['plan']}"),
                HumanMessage(content=f"User request:\n{state['user_input']}"),
                first_response,
                *tool_messages,
            ]
        )
    else:
        final_response = first_response

    return {
        "response": str(final_response.content),
        "messages": [first_response, *tool_messages, final_response],
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("golden_profile", golden_profile_node)
    graph.add_node("batch_analysis", batch_analysis_node)
    graph.add_node("report", report_node)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_edge(START, "golden_profile")
    graph.add_edge("golden_profile", "batch_analysis")
    graph.add_edge("batch_analysis", "report")
    graph.add_edge("report", "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", END)
    return graph.compile()


def run_agent(user_input: str) -> dict[str, Any]:
    app = build_graph()
    result = app.invoke(
        {
            "user_input": user_input,
            "plan": "",
            "response": "",
            "golden_profile": {"meta": {"case": "A", "status": "not_computed"}, "rows": []},
            "batch_analysis": {},
            "report": "",
            "messages": [],
        }
    )
    return {
        "plan": result.get("plan", ""),
        "response": result.get("response", ""),
        "batch_analysis": result.get("batch_analysis", {}),
        "report": result.get("report", ""),
    }
