"""Bonus 3 — Multi-reviewer fan-out: escalate the same questions to two reviewers in parallel and wait for both.

    uv run python -m bonus.multi_reviewer --pr <url> --reviewers alice,bob                 # answer in the terminal
    uv run python -m bonus.multi_reviewer --pr <url> --reviewers alice,bob --answers a.json # scripted, by keyword

Graph (nodes reused from Exercise 4 except the ones marked *):

    fetch_pr → analyze → route* ─┬─ auto_approve
                                 ├─ human_approval → (revise) → commit
                                 └─ prepare_questions* ══Send══> ask_reviewer* (one branch per reviewer)
                                                                      ║ both branches interrupt()
                                                      merge_reviews* <╝ runs only after every branch resumed
                                                            → synthesize → commit

Each `ask_reviewer` branch is its own task with its own interrupt and its own audit rows (reviewer_id = that
reviewer). All interrupts are pending at once; the caller resumes them together with
`Command(resume={interrupt_id: answers, ...})`. The join is LangGraph's: `merge_reviews` runs in the next
superstep, after all Send branches finished, and the `reviews` list is merged with an `operator.add` reducer.

--escalate-below overrides ESCALATE_THRESHOLD for this run only (the README FAQ suggests raising it to 0.70
when the model is not unsure enough to escalate). Dry run by default; --post really posts the comment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import time
import uuid
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path
from common.llm import get_llm
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD, AuditEntry, ReviewState, risk_level_for
from exercises import exercise_4_audit as ex4


console = Console()


class MultiReviewState(ReviewState, total=False):
    reviewers: list[str]
    escalate_below: float
    reviewer: str                                       # set only inside one Send branch
    reviews: Annotated[list[dict], operator.add]        # one {"reviewer", "answers"} per branch
    reviewed_by: str                                    # audit reviewer_id for synthesize/commit


async def node_route(state):
    """Exercise 4's routing, with the escalate threshold taken from the state (CLI --escalate-below)."""
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    below = state.get("escalate_below", ESCALATE_THRESHOLD)
    decision = ("auto_approve" if c >= AUTO_APPROVE_THRESHOLD
                else "escalate" if c < below else "human_approval")
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%}, escalate < {below:.0%})")
    await ex4.audit(state, AuditEntry(
        agent_id=ex4.AGENT_ID, action="route", confidence=c, risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "escalate": "escalate", "human_approval": "pending"}[decision],
        reason=f"route={decision} (auto >= {AUTO_APPROVE_THRESHOLD:.2f}, escalate < {below:.2f}); "
               f"reviewers={','.join(state['reviewers'])}",
        execution_time_ms=ex4._ms(t0),
    ))
    return {"decision": decision}


async def node_prepare_questions(state):
    """Make sure there are questions before fanning out (Exercise 4 only writes them below 0.58)."""
    a = state["analysis"]
    if a.escalation_questions:
        return {}
    console.print("[cyan]→ prepare_questions[/cyan]")
    qs = await get_llm().with_structured_output(ex4.EscalationQuestions).ainvoke([
        {"role": "system", "content": ex4.QUESTIONS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}\n\n"
                                    f"Your review so far:\n{a.model_dump_json(indent=2)}"},
    ])
    return {"analysis": a.model_copy(update={"escalation_questions": qs.questions})}


def fan_out(state):
    # Same questions, one parallel branch per reviewer. Each Send carries the state that branch needs.
    return [Send("ask_reviewer", {**state, "reviewer": r}) for r in state["reviewers"]]


async def node_ask_reviewer(state):
    t0 = time.monotonic()
    a, who = state["analysis"], state["reviewer"]
    entry = dict(agent_id=ex4.AGENT_ID, action="escalate", confidence=a.confidence,
                 risk_level=risk_level_for(a.confidence), decision="escalate")
    await ex4.audit(state, AuditEntry(**entry, reason=f"asked {who} {len(a.escalation_questions)} question(s)",
                                      execution_time_ms=ex4._ms(t0)))
    answers = interrupt({
        "kind": "escalation", "reviewer": who, "pr_url": state["pr_url"], "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning, "summary": a.summary,
        "risk_factors": a.risk_factors, "questions": a.escalation_questions,
    })
    await ex4.audit(state, AuditEntry(**entry, reviewer_id=who, reason=f"{who} answered {len(answers)} question(s)",
                                      execution_time_ms=ex4._ms(t0)))
    return {"reviews": [{"reviewer": who, "answers": answers}]}


async def node_merge_reviews(state):
    """Join: runs once, after every reviewer branch has resumed. Folds both answers into one Q&A per question."""
    console.print(f"[cyan]→ merge_reviews[/cyan]  {len(state['reviews'])} reviewer(s) answered")
    t0 = time.monotonic()
    a = state["analysis"]
    merged = {q: "\n".join(f"[{r['reviewer']}] {r['answers'].get(q, '(no answer)')}" for r in state["reviews"])
              for q in a.escalation_questions}
    await ex4.audit(state, AuditEntry(
        agent_id=ex4.AGENT_ID, action="merge_reviews", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), reviewer_id=",".join(r["reviewer"] for r in state["reviews"]),
        decision="escalate", reason=f"merged answers from {len(state['reviews'])} reviewers",
        execution_time_ms=ex4._ms(t0),
    ))
    # ex4.node_synthesize reads escalation_answers; disagreements between reviewers stay visible to the LLM.
    return {"escalation_answers": merged, "reviewed_by": ",".join(r["reviewer"] for r in state["reviews"])}


def build_graph(checkpointer):
    g = StateGraph(MultiReviewState)
    for name, fn in [
        ("fetch_pr", ex4.node_fetch_pr), ("analyze", ex4.node_analyze), ("route", node_route),
        ("auto_approve", ex4.node_auto_approve), ("human_approval", ex4.node_human_approval),
        ("revise", ex4.node_revise), ("commit", ex4.node_commit),
        ("prepare_questions", node_prepare_questions), ("ask_reviewer", node_ask_reviewer),
        ("merge_reviews", node_merge_reviews), ("synthesize", ex4.node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "prepare_questions"},
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges(
        "human_approval", lambda s: "revise" if s.get("human_choice") == "edit" else "commit",
        {"revise": "revise", "commit": "commit"},
    )
    g.add_edge("revise", "commit")
    g.add_conditional_edges("prepare_questions", fan_out, ["ask_reviewer"])
    g.add_edge("ask_reviewer", "merge_reviews")
    g.add_edge("merge_reviews", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def _answer(payload: dict, scripted: dict | None) -> dict:
    who, qs = payload["reviewer"], payload["questions"]
    console.print(Panel.fit(f"{payload['summary']}\n\nRisks: {', '.join(payload['risk_factors'])}",
                            title=f"escalation → {who} (confidence {payload['confidence']:.0%})",
                            border_style="yellow"))
    if scripted is not None:
        # The LLM writes 2-4 questions, so scripted answers are matched by keyword. No match → stop, never guess.
        out = {}
        for q in qs:
            ans = next((a for keys, a in scripted[who] if any(k in q.lower() for k in keys)), None)
            if ans is None:
                raise SystemExit(f"No scripted answer from {who} for: {q}")
            console.print(f"Q: {q}\nA ({who}): {ans}")
            out[q] = ans
        return out
    return {q: console.input(f"Q: {q}\nA ({who}): ").strip() for q in qs}


async def run(pr_url: str, reviewers: list[str], escalate_below: float, scripted: dict | None) -> None:
    thread_id = str(uuid.uuid4())
    console.rule("[bold]Bonus 3 — multi-reviewer fan-out[/bold]")
    console.print(f"[dim]PR: {pr_url}\nthread_id = {thread_id}\nreviewers = {reviewers}[/dim]\n")
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id, "reviewers": reviewers,
                                    "escalate_below": escalate_below, "reviews": []}, cfg)
        while "__interrupt__" in result:
            pending = result["__interrupt__"]
            console.print(f"\n[bold]{len(pending)} interrupt(s) pending at once[/bold]: "
                          + ", ".join(i.value.get("reviewer", i.value["kind"]) for i in pending))
            resume = {}
            for i in pending:
                if i.value["kind"] == "approval_request":
                    raise SystemExit("Routed to single-reviewer approval, not escalation — rerun with a higher "
                                     "--escalate-below or use Exercise 4 for approvals.")
                resume[i.id] = _answer(i.value, scripted)
            result = await app.ainvoke(Command(resume=resume), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--reviewers", default="reviewer-a,reviewer-b", help="Comma-separated reviewer ids")
    p.add_argument("--escalate-below", type=float, default=ESCALATE_THRESHOLD)
    p.add_argument("--answers", help='JSON file {reviewer: [[["keyword", ...], "answer"], ...]}')
    p.add_argument("--post", action="store_true", help="Really post the comment (default: dry run)")
    args = p.parse_args()
    if not args.post:
        os.environ["HITL_DRY_RUN"] = "1"
    scripted = json.loads(open(args.answers, encoding="utf-8").read()) if args.answers else None
    asyncio.run(run(args.pr, [r.strip() for r in args.reviewers.split(",") if r.strip()],
                    args.escalate_below, scripted))


if __name__ == "__main__":
    main()
