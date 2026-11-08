"""
alert_dispatch_dag.py
=====================
Weekly alert for one city: email the CRITICAL tracts with their drivers
and key numbers. Optionally generates policy briefings first.

Started by spatial_transform_<city> with {"week_start": "YYYY-MM-DD"}.

Tasks:
    resolve_week        the Monday to report on
    generate_briefings  model briefings for HIGH and CRITICAL tracts. Skipped
                        unless triggered by hand with {"generate_briefings": true},
                        so scheduled weeks make no model calls
    build_digest        critical tracts, drivers, numbers, NEW flags, and
                        briefing summaries when the week has them; saves
                        HTML and CSV under data/alerts/<city>/<week>/
    send_digest         emails the digest through the smtp_default connection
                        to ALERT_EMAIL_TO. Skipped if no recipients are set
                        or the run has {"send_email": false}

Briefing runs keep a checkpoint per tract in data/briefing_runs/<week>/, so
a retry or rerun only pays for tracts not done yet. The week is written to
the warehouse only if every tract succeeded.
"""

from __future__ import annotations

from datetime import timedelta

from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowSkipException
from chbindex_common import CITIES, DEFAULT_ARGS, ROOT, WEEK_PATTERN, project_config


def build_alert_dag(city: str):

    @dag(
        dag_id=f"alert_dispatch_{city}",
        description=f"Critical-tract email and optional briefings for {city}",
        schedule=None,  # triggered by spatial_transform_<city>
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        params={
            "week_start": Param(type="string", pattern=WEEK_PATTERN, description="Monday of the week, YYYY-MM-DD."),
            "generate_briefings": Param(
                False, type="boolean",
                description="Generate model briefings first (costs API credit; about 1 to 2 USD for LA).",
            ),
            "send_email": Param(True, type="boolean", description="Email the digest to ALERT_EMAIL_TO."),
        },
        tags=["chbindex", "alerts", city],
    )
    def alert_dispatch():

        @task
        def resolve_week() -> str:
            from datetime import date

            from src.utils.weeks import monday_of

            return monday_of(date.fromisoformat(get_current_context()["params"]["week_start"])).isoformat()

        # Several hundred model calls take longer than the default timeout
        @task(execution_timeout=timedelta(hours=3))
        def generate_briefings(week: str) -> dict:
            if not get_current_context()["params"]["generate_briefings"]:
                raise AirflowSkipException("Briefings not requested for this run")

            from datetime import date

            from src.rag.briefing import BriefingGenerator
            from src.rag.briefing_runner import commit_week, run_week
            from src.rag.risk_inputs import build_tract_risks

            config = project_config(city)
            risks = build_tract_risks(config, week)
            checkpoint = ROOT / "data" / "briefing_runs" / week / "full_checkpoint.jsonl"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            briefings, failures = run_week(risks, BriefingGenerator(config), checkpoint)
            commit_week(briefings, failures, date.fromisoformat(week), config)  # raises if any failed
            return {"briefings": len(briefings)}

        # Runs whether briefings were generated or skipped, but not if they failed
        @task(trigger_rule="none_failed")
        def build_digest(week: str) -> dict:
            from src.alerts.digest import build_digest as collect
            from src.alerts.digest import render_html, render_subject, write_csv

            config = project_config(city)
            digest = collect(config, week)
            out_dir = ROOT / "data" / "alerts" / city / week
            out_dir.mkdir(parents=True, exist_ok=True)
            html_path = out_dir / "digest.html"
            html_path.write_text(render_html(digest, top_n=config["alerts"]["top_n"]), encoding="utf-8")
            csv_path = write_csv(digest, out_dir / f"critical_tracts_{week}.csv")
            return {
                "subject": render_subject(digest),
                "html_path": str(html_path),
                "csv_path": str(csv_path),
                "critical": len(digest.critical),
                "new": len(digest.newly_critical),
            }

        @task
        def send_digest(digest: dict) -> list[str]:
            if not get_current_context()["params"]["send_email"]:
                raise AirflowSkipException("Email turned off for this run")

            from airflow.providers.smtp.hooks.smtp import SmtpHook

            recipients = [a.strip() for a in project_config(city)["secrets"]["ALERT_EMAIL_TO"].split(",") if a.strip()]
            if not recipients:
                raise AirflowSkipException(f"ALERT_EMAIL_TO is empty; digest saved at {digest['html_path']}")

            with open(digest["html_path"], encoding="utf-8") as f:
                html = f.read()
            with SmtpHook(smtp_conn_id="smtp_default") as hook:
                hook.send_email_smtp(
                    to=recipients,
                    subject=digest["subject"],
                    html_content=html,
                    files=[digest["csv_path"]],
                )
            return recipients

        week = resolve_week()
        briefings = generate_briefings(week)
        digest = build_digest(week)
        briefings >> digest
        send_digest(digest)

    return alert_dispatch()


for _city in CITIES:
    globals()[f"alert_dispatch_{_city}"] = build_alert_dag(_city)
