#!/usr/bin/env python3
"""Live smoke test of the Monitors tools (langchain-desearch 1.5.0).

Creates one monitor without a first run, then calls every free action once:
read, list, estimate, update (options merge, new webhook URL), pause, resume,
runs, items, webhook test, secret rotation and webhook removal, and deletes the
monitor at the end, also when a step fails. Prints one line per call: ``PASS
<name>``, ``FAIL <name>: <error>`` or ``SKIP <name>: <why>``. Exits 1 if any
call fails.

Everything it calls is free. ``--trigger`` also runs the monitor once and
waits for the run, which is billed (about $0.002-0.02). The webhook points at
https://example.com, which does not accept POSTs, so the test event is not
delivered; the step checks the answer's shape.

``--base-url`` points the tools at another deployment, such as
https://api-dev.desearch.ai, and ``--key-env`` names the variable that holds
its key. Keys are read from the environment or ``~/.config/desearch/e2e.env``.

    python scripts/e2e-monitors.py --base-url https://api-dev.desearch.ai --key-env DESEARCH_DEV_API_KEY

The tools call the routes over HTTP, so this needs no unreleased desearch-py.
Not run by CI.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# Run from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_desearch import (  # noqa: E402
    MonitorControlTool,
    MonitorCreateTool,
    MonitorEstimateTool,
    MonitorItemsTool,
    MonitorRunsTool,
    MonitorsTool,
    MonitorUpdateTool,
)

ERROR_PREFIX = "An error occurred while calling Desearch"
WEBHOOK_URL = "https://example.com/desearch-monitors-e2e"
NEW_WEBHOOK_URL = "https://example.com/desearch-monitors-e2e-2"


def _load_env_file() -> None:
    path = Path.home() / ".config" / "desearch" / "e2e.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value and not os.environ.get(key):
            os.environ[key] = value


def _one_line(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else f"{text[:limit]}..."


class Smoke:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run(
        self,
        name: str,
        call: Callable[[], Any],
        check: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> Any:
        """Run one call. ``check`` returns a problem, or None if the result is fine."""
        try:
            result = call()
            if isinstance(result, str) and result.startswith(ERROR_PREFIX):
                problem: Optional[str] = result
            else:
                problem = check(result) if check else None
        except Exception as exc:  # report and keep going
            result, problem = None, f"{type(exc).__name__}: {exc}"
        if problem:
            print(f"FAIL {name}: {_one_line(problem)}", flush=True)
            self.failed += 1
            return None
        print(f"PASS {name}", flush=True)
        self.passed += 1
        return result

    def skip(self, name: str, why: str) -> None:
        print(f"SKIP {name}: {why}", flush=True)
        self.skipped += 1


def _expect(**expected: Any) -> Callable[[Any], Optional[str]]:
    def check(result: Any) -> Optional[str]:
        if not isinstance(result, dict):
            return f"expected an object, got {_one_line(repr(result), 200)}"
        for key, value in expected.items():
            if value is ... and key not in result:
                return f"{key!r} is missing"
            if value is not ... and result.get(key) != value:
                return f"{key}={result.get(key)!r}, expected {value!r}"
        return None

    return check


def _has_list(key: str) -> Callable[[Any], Optional[str]]:
    def check(result: Any) -> Optional[str]:
        if not isinstance(result, dict) or not isinstance(result.get(key), list):
            return f"expected {{{key}: [...]}}, got {_one_line(repr(result), 200)}"
        return None

    return check


def check_monitors(smoke: Smoke, base_url: str, trigger: bool) -> None:
    tools = {
        "create": MonitorCreateTool(base_url=base_url),
        "estimate": MonitorEstimateTool(base_url=base_url),
        "monitors": MonitorsTool(base_url=base_url),
        "update": MonitorUpdateTool(base_url=base_url),
        "control": MonitorControlTool(base_url=base_url),
        "runs": MonitorRunsTool(base_url=base_url, poll_interval=5.0),
        "items": MonitorItemsTool(base_url=base_url),
    }
    smoke.run(
        "monitor_estimate_tool",
        lambda: tools["estimate"].invoke({"sources": ["web", "x"], "interval": "1d"}),
        _expect(max_run_cost_usd=..., runs_per_month=30, breakdown=...),
    )
    name = f"langchain e2e {int(time.time())}"
    created = smoke.run(
        "monitor_create_tool (run_now=false, webhook)",
        lambda: tools["create"].invoke(
            {
                "query": "Bittensor subnet launches and updates",
                "name": name,
                "criteria": ["Announces a new subnet, release or partnership"],
                "sources": ["web", "x", "reddit"],
                "interval": "7d",
                "x_count": 10,
                "x_lang": "en",
                "webhook_url": WEBHOOK_URL,
                "metadata": {"e2e": "langchain"},
                "run_now": False,
            }
        ),
        _expect(status="active", interval="7d", webhook_secret=..., max_run_cost_usd=...),
    )
    if not created:
        smoke.skip("the remaining monitor steps", "the monitor was not created")
        return
    monitor_id = created["monitor_id"]
    first_secret = created["webhook_secret"]
    try:
        smoke.run(
            "monitors_tool (one)",
            lambda: tools["monitors"].invoke({"monitor_id": monitor_id}),
            _expect(monitor_id=monitor_id, name=name, metadata={"e2e": "langchain"}),
        )

        def listed(result: Any) -> Optional[str]:
            problem = _has_list("monitors")(result)
            if problem:
                return problem
            ids = [m.get("monitor_id") for m in result["monitors"]]
            return None if monitor_id in ids else "the new monitor is not on the first page"

        smoke.run(
            "monitors_tool (list active)",
            lambda: tools["monitors"].invoke({"status": ["active"], "limit": 100}),
            listed,
        )

        def merged(result: Any) -> Optional[str]:
            x = (result.get("options") or {}).get("x") or {}
            if result.get("name") != f"{name} (renamed)":
                return f"name={result.get('name')!r}"
            if x.get("count") != 25 or x.get("lang") != "en":
                return f"x options were not merged: {x!r}"
            return None

        smoke.run(
            "monitor_update_tool (name, x_count merge)",
            lambda: tools["update"].invoke(
                {"monitor_id": monitor_id, "name": f"{name} (renamed)", "x_count": 25}
            ),
            merged,
        )

        def new_secret(result: Any) -> Optional[str]:
            secret = result.get("webhook_secret")
            if not secret or secret == first_secret:
                return "a new webhook URL did not return a new secret"
            if (result.get("webhook") or {}).get("url") != NEW_WEBHOOK_URL:
                return f"webhook={result.get('webhook')!r}"
            return None

        smoke.run(
            "monitor_update_tool (new webhook URL)",
            lambda: tools["update"].invoke(
                {"monitor_id": monitor_id, "webhook_url": NEW_WEBHOOK_URL}
            ),
            new_secret,
        )
        smoke.run(
            "monitor_control_tool pause",
            lambda: tools["control"].invoke({"monitor_id": monitor_id, "action": "pause"}),
            _expect(status="paused", paused_reason="user"),
        )
        smoke.run(
            "monitor_control_tool resume",
            lambda: tools["control"].invoke({"monitor_id": monitor_id, "action": "resume"}),
            _expect(status="active"),
        )
        smoke.run(
            "monitor_runs_tool (list)",
            lambda: tools["runs"].invoke({"monitor_id": monitor_id}),
            _has_list("runs"),
        )
        smoke.run(
            "monitor_items_tool",
            lambda: tools["items"].invoke({"monitor_id": monitor_id, "source": "x"}),
            _has_list("items"),
        )
        smoke.run(
            "monitor_items_tool (ainvoke)",
            lambda: asyncio.run(tools["items"].ainvoke({"monitor_id": monitor_id})),
            _has_list("items"),
        )
        smoke.run(
            "monitor_control_tool test_webhook",
            lambda: tools["control"].invoke(
                {"monitor_id": monitor_id, "action": "test_webhook"}
            ),
            _expect(delivered=..., duration_ms=..., next_step=...),
        )
        smoke.run(
            "monitor_control_tool rotate_webhook_secret",
            lambda: tools["control"].invoke(
                {"monitor_id": monitor_id, "action": "rotate_webhook_secret"}
            ),
            _expect(webhook_secret=...),
        )
        smoke.run(
            "monitor_update_tool (remove_webhook)",
            lambda: tools["update"].invoke({"monitor_id": monitor_id, "remove_webhook": True}),
            lambda result: "the webhook is still set" if result.get("webhook") else None,
        )
        if trigger:
            run = smoke.run(
                "monitor_control_tool trigger (billed)",
                lambda: tools["control"].invoke({"monitor_id": monitor_id, "action": "trigger"}),
                _expect(run_id=..., status="queued"),
            )
            if run:
                smoke.run(
                    "monitor_runs_tool (wait for the run)",
                    lambda: tools["runs"].invoke(
                        {"monitor_id": monitor_id, "run_id": run["run_id"], "timeout": 300}
                    ),
                    _expect(done=True, items=...),
                )
        else:
            smoke.skip("monitor_control_tool trigger", "billed; pass --trigger to run it")
    finally:
        smoke.run(
            "monitor_control_tool delete",
            lambda: tools["control"].invoke({"monitor_id": monitor_id, "action": "delete"}),
            _expect(monitor_id=monitor_id, deleted=True),
        )
    # Smoke.run treats an error as a failure; here the 404 is the expected answer.
    deleted = tools["monitors"].invoke({"monitor_id": monitor_id})
    if isinstance(deleted, str) and "HTTP 404" in deleted:
        print("PASS monitors_tool (deleted is 404)", flush=True)
        smoke.passed += 1
    else:
        print(f"FAIL monitors_tool (deleted is 404): {_one_line(repr(deleted), 200)}", flush=True)
        smoke.failed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DESEARCH_BASE_URL", "https://api.desearch.ai"),
        help="API base URL (default: $DESEARCH_BASE_URL or production)",
    )
    parser.add_argument(
        "--key-env",
        default="DESEARCH_API_KEY",
        help="environment variable that holds the API key (default: DESEARCH_API_KEY)",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="also run the monitor once and wait for the run (billed)",
    )
    args = parser.parse_args()
    _load_env_file()
    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        print(f"{args.key_env} is not set.", file=sys.stderr)
        return 2
    # The tools read DESEARCH_API_KEY; each tool gets the base URL under test.
    os.environ["DESEARCH_API_KEY"] = api_key
    smoke = Smoke()
    check_monitors(smoke, args.base_url, args.trigger)
    print(
        f"{smoke.passed} passed, {smoke.failed} failed, {smoke.skipped} skipped "
        f"on {args.base_url}"
    )
    return 1 if smoke.failed else 0


if __name__ == "__main__":
    sys.exit(main())
