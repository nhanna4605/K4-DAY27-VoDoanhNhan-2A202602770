"""Bonus 1 — Time-travel: fork a finished review from its HITL checkpoint with a different reviewer answer.

    uv run python -m bonus.time_travel --thread <id>                       # list checkpoints only
    uv run python -m bonus.time_travel --thread <id> --choice reject        # approval thread: answer differently
    uv run python -m bonus.time_travel --thread <id> --answers alt.json     # escalation thread: JSON list of answers

How it works:
1. `aget_state_history()` lists every checkpoint of the thread (newest first).
2. The fork point is the checkpoint whose next node is the HITL node (`human_approval` / `escalate`).
3. `aupdate_state(fork_point, {"thread_id": "<id>:fork-N"}, as_node="route")` creates a new checkpoint branching
   off that point. Only the audit label changes, so the fork's audit rows replay separately
   (`audit.replay --thread <id>:fork-N`) while the checkpointer keeps both branches under the same thread.
4. Running the graph from the fork re-asks the same question (interrupt), we resume with the alternative answer,
   and the outcome is compared with the original branch.

The fork never posts to GitHub unless --post is given (HITL_DRY_RUN): the original branch already did.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


console = Console()
HITL_NODES = ("human_approval", "escalate")


def _outcome(values: dict) -> dict:
    a = values.get("analysis")
    return {
        "reviewer input": (values.get("human_choice") and
                           f"{values['human_choice']}: {values.get('human_feedback') or '-'}")
                          or "\n".join(f"- {v}" for v in (values.get("escalation_answers") or {}).values()) or "-",
        "final confidence": f"{a.confidence:.0%}" if a else "-",
        "comments": str(len(a.comments)) if a else "-",
        "blockers/issues": str(sum(c.severity in ("blocker", "issue") for c in a.comments)) if a else "-",
        "final_action": values.get("final_action") or "-",
    }


async def _next_fork_label(thread_id: str) -> str:
    async with db_conn() as conn:
        async with conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM audit_events WHERE thread_id LIKE ?", (f"{thread_id}:fork-%",),
        ) as cur:
            n = (await cur.fetchone())[0]
    return f"{thread_id}:fork-{n + 1}"


async def run(thread_id: str, choice: str | None, feedback: str, answers: list[str] | None) -> None:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        history = [s async for s in app.aget_state_history(cfg)]
        if not history:
            raise SystemExit(f"No checkpoints for thread {thread_id}")

        t = Table(title=f"Checkpoints of {thread_id} (newest first)")
        for col in ("step", "checkpoint", "source", "next", "decision", "final_action"):
            t.add_column(col)
        for s in history:
            t.add_row(str(s.metadata.get("step")), s.config["configurable"]["checkpoint_id"][-8:],
                      str(s.metadata.get("source")), ",".join(s.next) or "END",
                      str(s.values.get("decision", "")), str(s.values.get("final_action", "")))
        console.print(t)

        forks = [s for s in history if s.next and s.next[0] in HITL_NODES]
        if not forks:
            raise SystemExit("This thread never reached a HITL node — nothing to time-travel to.")
        fork_point = forks[-1]          # oldest = the original pause (later ones are earlier forks)
        node = fork_point.next[0]
        console.print(f"\nFork point: checkpoint …{fork_point.config['configurable']['checkpoint_id'][-8:]} "
                      f"(step {fork_point.metadata.get('step')}), next node = [bold]{node}[/bold]")
        if choice is None and answers is None:
            return

        original = _outcome(history[0].values)
        label = await _next_fork_label(thread_id)
        fork_cfg = await app.aupdate_state(fork_point.config, {"thread_id": label}, as_node="route")
        console.print(f"Created fork [bold]{label}[/bold] → checkpoint …{fork_cfg['configurable']['checkpoint_id'][-8:]}\n")

        result = await app.ainvoke(None, fork_cfg)          # re-runs the HITL node → same question again
        latest = {"configurable": {"thread_id": thread_id}}  # the fork head is now the thread's latest checkpoint
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            if payload["kind"] == "approval_request":
                if choice is None:
                    raise SystemExit("Fork paused at an approval request: pass --choice approve|reject|edit")
                resume = {"choice": choice, "feedback": feedback}
            else:
                if answers is None:
                    raise SystemExit("Fork paused at an escalation: pass --answers <json list>")
                qs = payload["questions"]
                if len(answers) != len(qs):
                    raise SystemExit(f"--answers has {len(answers)} answers but the escalation asks {len(qs)}:\n"
                                     + "\n".join(qs))
                resume = dict(zip(qs, answers))
            result = await app.ainvoke(Command(resume=resume), latest)

        forked = _outcome(result)
        cmp = Table(title="Original branch vs fork")
        cmp.add_column("")
        cmp.add_column(f"original ({thread_id[:8]})", overflow="fold")
        cmp.add_column(label.split(":")[-1], overflow="fold")
        for k in original:
            cmp.add_row(k, original[k], forked[k])
        console.print(cmp)
        console.print(f"\n[dim]Audit of the fork:[/dim] uv run python -m audit.replay --thread {label}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--thread", required=True)
    p.add_argument("--choice", choices=("approve", "reject", "edit"))
    p.add_argument("--feedback", default="")
    p.add_argument("--answers", help="JSON file: list of answers, in question order")
    p.add_argument("--post", action="store_true", help="Really post the fork's comment (default: dry run)")
    args = p.parse_args()
    if not args.post:
        os.environ["HITL_DRY_RUN"] = "1"
    answers = json.loads(open(args.answers, encoding="utf-8").read()) if args.answers else None
    asyncio.run(run(args.thread, args.choice, args.feedback, answers))


if __name__ == "__main__":
    main()
