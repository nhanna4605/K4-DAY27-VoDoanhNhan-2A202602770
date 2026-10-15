"""Bonus 2 — Confidence calibration from the audit trail.

    uv run python -m bonus.calibration [--pr <url>]

For every session a human decided on (approve / reject / edit), compare the model's confidence at `analyze`
with what the reviewer actually did. "Approved as-is" is the outcome the confidence claims to predict, so
`edit` and `reject` both count as "not approved".

- AVG(confidence) vs approval rate, overall and per routing bucket
- gap = avg confidence − approval rate  (> 0: over-confident, < 0: under-confident)
- Brier score = mean((confidence − approved)²), 0 is perfect, 0.25 is a coin flip at 50%
- escalated sessions: confidence before vs after the reviewer's answers
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from common.db import db_conn
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD


console = Console()
MIN_SESSIONS = 10   # below this, report the numbers but do not draw a conclusion

# Last human decision per session. The human_approval row carries the confidence the reviewer was shown
# (the analyze confidence), so fork threads (bonus 1), which have no analyze row, are counted too.
HUMAN_SQL = """
SELECT thread_id, pr_url, confidence, decision
  FROM audit_events
 WHERE id IN (SELECT MAX(id) FROM audit_events
               WHERE action = 'human_approval' AND decision IN ('approve', 'reject', 'edit')
               GROUP BY thread_id)
   AND (? IS NULL OR pr_url = ?)
 ORDER BY id
"""

ESCALATION_SQL = """
SELECT e.thread_id, e.pr_url,
       (SELECT confidence FROM audit_events WHERE thread_id = e.thread_id AND action = 'escalate' ORDER BY id LIMIT 1),
       (SELECT confidence FROM audit_events WHERE thread_id = e.thread_id AND action = 'synthesize'
         ORDER BY id DESC LIMIT 1)
  FROM (SELECT DISTINCT thread_id, pr_url FROM audit_events WHERE action = 'synthesize') e
 WHERE (? IS NULL OR e.pr_url = ?)
"""


def _bucket(c: float) -> str:
    if c >= AUTO_APPROVE_THRESHOLD:
        return f"auto (>= {AUTO_APPROVE_THRESHOLD:.0%})"
    if c < ESCALATE_THRESHOLD:
        return f"escalate (< {ESCALATE_THRESHOLD:.0%})"
    return "human approval"


def _verdict(gap: float, n: int) -> str:
    if n < MIN_SESSIONS:
        lean = "over" if gap > 0 else "under"
        return (f"only {n} human-decided session(s) — too few to conclude "
                f"(so far it leans {lean}-confident, gap {gap:+.0%})")
    if gap > 0.10:
        return "over-confident: it claims more than reviewers accept"
    if gap < -0.10:
        return "under-confident: reviewers accept more than it claims"
    return "roughly calibrated (|gap| <= 10 points)"


async def run(pr: str | None) -> None:
    async with db_conn() as conn:
        async with conn.execute(HUMAN_SQL, (pr, pr)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        async with conn.execute(ESCALATION_SQL, (pr, pr)) as cur:
            esc = list(await cur.fetchall())

    console.rule("[bold]Bonus 2 — confidence calibration[/bold]")
    if not rows:
        console.print("No human-decided sessions in audit_events yet.")
    else:
        t = Table(title="Human-decided sessions")
        for col in ("thread", "pr", "confidence", "bucket", "decision"):
            t.add_column(col)
        for r in rows:
            t.add_row(r["thread_id"][:8] + (" (fork)" if ":fork-" in r["thread_id"] else ""),
                      r["pr_url"].split("github.com/")[-1], f"{r['confidence']:.0%}",
                      _bucket(r["confidence"]), r["decision"])
        console.print(t)

        def stats(rs):
            n = len(rs)
            avg = sum(r["confidence"] for r in rs) / n
            rate = sum(r["decision"] == "approve" for r in rs) / n
            brier = sum((r["confidence"] - (r["decision"] == "approve")) ** 2 for r in rs) / n
            return n, avg, rate, brier

        s = Table(title="AVG(confidence) vs approval rate")
        for col in ("group", "n", "AVG(confidence)", "approved as-is", "gap", "Brier"):
            s.add_column(col)
        groups = {"all": rows}
        for r in rows:
            groups.setdefault(_bucket(r["confidence"]), []).append(r)
        for name, rs in groups.items():
            n, avg, rate, brier = stats(rs)
            s.add_row(name, str(n), f"{avg:.0%}", f"{rate:.0%} ({sum(r['decision'] == 'approve' for r in rs)}/{n})",
                      f"{avg - rate:+.0%}", f"{brier:.3f}")
        console.print(s)
        n, avg, rate, _ = stats(rows)
        console.print(f"Verdict: [bold]{_verdict(avg - rate, n)}[/bold]")

    if esc:
        e = Table(title="Escalated sessions — confidence before vs after the reviewer's answers")
        for col in ("thread", "pr", "before", "after", "change"):
            e.add_column(col)
        for thread, url, before, after in esc:
            e.add_row(thread[:8] + (" (fork)" if ":fork-" in thread else ""), url.split("github.com/")[-1],
                      f"{before:.0%}" if before is not None else "-", f"{after:.0%}",
                      f"{after - before:+.0%}" if before is not None else "-")
        console.print(e)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", help="Only sessions for this PR URL")
    args = p.parse_args()
    asyncio.run(run(args.pr))


if __name__ == "__main__":
    main()
