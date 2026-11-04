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
delivered; the step checks the answer's shape. ``--with-accounts`` also creates
(and deletes) a monitor with the TikTok source and watched accounts on X,
Reddit, YouTube and TikTok (desearch-public-api#635): the names come back
cleaned, a bad name is a 422, and an update keeps the accounts it does not
change. It then creates (and deletes) a monitor with the Instagram, Facebook
and Meta ads sources and Threads accounts (desearch-public-api#644): the names
come back cleaned, instagram or facebook without accounts is a 422 on create
and on update, the estimate prices each source, a fresh webhook has
``last_delivery`` null, and items can be read by source meta_ads. Also free:
every monitor is created with ``run_now: false`` and never triggered.

``--base-url`` points the tools at another deployment, such as
https://api-dev.desearch.ai, and ``--key-env`` names the variable that holds
its key. Keys are read from the environment or ``~/.config/desearch/e2e.env``.

    python scripts/e2e-monitors.py --base-url https://api-dev.desearch.ai --key-env DESEARCH_DEV_API_KEY [--with-accounts]

The tools call the routes over HTTP, so this needs no unreleased desearch-py.
Not run by CI.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


def _expect_422(
    smoke: Smoke, name: str, call: Callable[[], Any], detail: str = ""
) -> Any:
    """Run a call whose expected answer is HTTP 422 (``Smoke.run`` counts errors as failures).

    ``detail`` must appear in the error, so a 422 for another field does not pass.
    """
    try:
        result = call()
    except Exception as exc:  # report and keep going
        result = f"{type(exc).__name__}: {exc}"
    if isinstance(result, str) and "HTTP 422" in result and detail in result:
        print(f"PASS {name}", flush=True)
        smoke.passed += 1
    else:
        # A dict here is a monitor that should not exist; show only its id.
        shown = (
            f"no error, monitor_id={result.get('monitor_id')!r}"
            if isinstance(result, dict)
            else repr(result)
        )
        print(f"FAIL {name}: {_one_line(shown, 200)}", flush=True)
        smoke.failed += 1
    return result


def _recording(call: Callable[[], Any], created: List[str]) -> Callable[[], Any]:
    """``call``, noting the monitor it creates before any check runs.

    ``Smoke.run`` returns None when a check fails, so the id is kept here to delete
    the monitor even then.
    """

    def wrapped() -> Any:
        result = call()
        if isinstance(result, dict) and result.get("monitor_id"):
            created.append(result["monitor_id"])
        return result

    return wrapped


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


# X usernames are 1-15 characters; the API answers 422 for a longer one.
X_USERNAME_MAX = 15
WATCHED = {
    "x_accounts": ["@OpenAI", "https://x.com/AnthropicAI"],
    "reddit_subreddits": ["r/LocalLLaMA"],
    "youtube_channels": ["https://www.youtube.com/@OpenAI"],
    "tiktok_accounts": ["@openai"],
    "tiktok_count": 20,
}
CLEANED = {
    "x": ["OpenAI", "AnthropicAI"],
    "reddit": ["LocalLLaMA"],
    "youtube": ["@OpenAI"],
    "tiktok": ["openai"],
}
ACCOUNT_FIELDS = {"x": "accounts", "reddit": "subreddits", "youtube": "channels", "tiktok": "accounts"}
assert all(len(name) <= X_USERNAME_MAX for name in CLEANED["x"]), "X usernames are 1-15 characters"


def _watched(result: Any) -> Dict[str, Any]:
    options = result.get("options") or {}
    return {
        source: (options.get(source) or {}).get(field) for source, field in ACCOUNT_FIELDS.items()
    }


def check_watched_accounts(smoke: Smoke, base_url: str) -> None:
    create = MonitorCreateTool(base_url=base_url)
    estimate = MonitorEstimateTool(base_url=base_url)
    update = MonitorUpdateTool(base_url=base_url)
    control = MonitorControlTool(base_url=base_url)
    items = MonitorItemsTool(base_url=base_url)
    sources = ["x", "reddit", "youtube", "tiktok"]

    def priced_tiktok(result: Any) -> Optional[str]:
        breakdown = result.get("breakdown") if isinstance(result, dict) else None
        return None if isinstance(breakdown, dict) and "tiktok" in breakdown else f"breakdown={breakdown!r}"

    smoke.run(
        "monitor_estimate_tool (tiktok + watched accounts)",
        lambda: estimate.invoke({"sources": sources, **WATCHED}),
        priced_tiktok,
    )
    _expect_422(
        smoke,
        "monitor_estimate_tool (bad X username is 422)",
        lambda: estimate.invoke({"sources": ["x"], "x_accounts": ["not a name!"]}),
    )
    _expect_422(
        smoke,
        f"monitor_estimate_tool ({X_USERNAME_MAX + 1}-character X username is 422)",
        lambda: estimate.invoke({"sources": ["x"], "x_accounts": ["a" * (X_USERNAME_MAX + 1)]}),
    )

    def cleaned(result: Any) -> Optional[str]:
        if result.get("sources") != sources:
            return f"sources={result.get('sources')!r}"
        if _watched(result) != CLEANED:
            return f"names={_watched(result)!r}"
        return None

    made: List[str] = []
    created = smoke.run(
        "monitor_create_tool (tiktok + watched accounts)",
        _recording(
            lambda: create.invoke(
                {
                    "query": "OpenAI announcements",
                    "name": f"langchain e2e accounts {int(time.time())}",
                    "sources": sources,
                    "interval": "7d",
                    "min_score": 0,
                    "run_now": False,
                    **WATCHED,
                }
            ),
            made,
        ),
        cleaned,
    )
    try:
        if not created:
            why = "its create check failed" if made else "the monitor was not created"
            smoke.skip("the remaining watched-accounts steps", why)
            return
        monitor_id = created["monitor_id"]

        def kept(result: Any) -> Optional[str]:
            tiktok = (result.get("options") or {}).get("tiktok")
            if tiktok != {"count": 30, "accounts": ["openai"]}:
                return f"tiktok={tiktok!r}"
            return None if _watched(result) == CLEANED else f"names={_watched(result)!r}"

        smoke.run(
            "monitor_update_tool (tiktok_count keeps every account)",
            lambda: update.invoke({"monitor_id": monitor_id, "tiktok_count": 30}),
            kept,
        )
        smoke.run(
            "monitor_items_tool (source tiktok)",
            lambda: items.invoke({"monitor_id": monitor_id, "source": "tiktok"}),
            _has_list("items"),
        )
    finally:
        for made_id in made:
            smoke.run(
                "monitor_control_tool delete (accounts)",
                lambda made_id=made_id: control.invoke({"monitor_id": made_id, "action": "delete"}),
                _expect(monitor_id=made_id, deleted=True),
            )


# Instagram and Facebook read only watched accounts; Meta ads search the Ad Library by
# keyword, so the query is a brand rather than a sentence.
SOCIAL_SOURCES = ["instagram", "facebook", "meta_ads", "threads"]
SOCIAL = {
    "instagram_accounts": ["@openai", "https://www.instagram.com/nasa/"],
    "facebook_pages": [
        "https://www.facebook.com/NASA",
        "https://www.facebook.com/profile.php?id=100064869009429",
    ],
    "threads_accounts": ["@zuck"],
    "meta_ads_count": 20,
}
SOCIAL_CLEANED = {
    "instagram": {"accounts": ["openai", "nasa"]},
    "facebook": {"pages": ["NASA", "100064869009429"]},
    "meta_ads": {"count": 20},
    "threads": {"accounts": ["zuck"]},
}
# Most a run of SOCIAL can cost per source (desearch-public-api#644): an Instagram account
# $0.02484, a Facebook page $0.0207, Meta ads $0.00207 per record, a Threads account
# $0.00207.
SOCIAL_PRICES = {
    "instagram": 2 * 0.02484,
    "facebook": 2 * 0.0207,
    "meta_ads": 20 * 0.00207,
    "threads": 0.00207,
}
ACCOUNT_DETAIL = {"instagram": "options.instagram.accounts", "facebook": "options.facebook.pages"}


def _social_options(result: Any) -> Dict[str, Any]:
    options = result.get("options") or {}
    return {source: options.get(source) for source in SOCIAL_CLEANED}


def _no_delivery_yet(result: Any) -> Optional[str]:
    webhook = result.get("webhook") or {}
    if not webhook.get("url"):
        return f"webhook={webhook!r}"
    if "last_delivery" not in webhook:
        return "webhook.last_delivery is missing"
    if webhook["last_delivery"] is not None:
        return f"last_delivery={webhook['last_delivery']!r}, expected null"
    return None


def check_social_sources(smoke: Smoke, base_url: str) -> None:
    create = MonitorCreateTool(base_url=base_url)
    estimate = MonitorEstimateTool(base_url=base_url)
    monitors = MonitorsTool(base_url=base_url)
    update = MonitorUpdateTool(base_url=base_url)
    control = MonitorControlTool(base_url=base_url)
    items = MonitorItemsTool(base_url=base_url)
    # Every monitor this creates, deleted at the end, also one a 422 check should not have made.
    created: List[str] = []

    def priced(result: Any) -> Optional[str]:
        breakdown = result.get("breakdown") if isinstance(result, dict) else None
        if not isinstance(breakdown, dict) or "screening" not in breakdown:
            return f"breakdown={breakdown!r}"
        for source, price in SOCIAL_PRICES.items():
            value = breakdown.get(source)
            if not isinstance(value, (int, float)) or abs(value - price) > 1e-6:
                return f"breakdown[{source!r}]={value!r}, expected {price:.6f}"
        return None

    try:
        smoke.run(
            "monitor_estimate_tool (instagram, facebook, meta_ads, threads accounts)",
            lambda: estimate.invoke({"sources": SOCIAL_SOURCES, **SOCIAL}),
            priced,
        )
        for source, detail in ACCOUNT_DETAIL.items():
            _expect_422(
                smoke,
                f"monitor_create_tool ({source} without accounts is 422)",
                _recording(
                    lambda source=source: create.invoke(
                        {"query": "NASA launches", "sources": [source], "run_now": False}
                    ),
                    created,
                ),
                detail,
            )

        def cleaned(result: Any) -> Optional[str]:
            if result.get("sources") != SOCIAL_SOURCES:
                return f"sources={result.get('sources')!r}"
            if _social_options(result) != SOCIAL_CLEANED:
                return f"options={_social_options(result)!r}"
            return _no_delivery_yet(result)

        before = len(created)
        monitor = smoke.run(
            "monitor_create_tool (instagram, facebook, meta_ads, threads accounts)",
            _recording(
                lambda: create.invoke(
                    {
                        "query": "ChatGPT",
                        "name": f"langchain e2e social {int(time.time())}",
                        "sources": SOCIAL_SOURCES,
                        "interval": "7d",
                        "webhook_url": WEBHOOK_URL,
                        "run_now": False,
                        **SOCIAL,
                    }
                ),
                created,
            ),
            cleaned,
        )
        if not monitor:
            why = "its create check failed" if len(created) > before else "the monitor was not created"
            smoke.skip("the remaining social-source steps", why)
            return
        monitor_id = monitor["monitor_id"]
        smoke.run(
            "monitors_tool (webhook.last_delivery is null before a delivery)",
            lambda: monitors.invoke({"monitor_id": monitor_id}),
            _no_delivery_yet,
        )
        for source, field in (("instagram", "instagram_accounts"), ("facebook", "facebook_pages")):
            _expect_422(
                smoke,
                f"monitor_update_tool ({field}=[] is 422)",
                lambda field=field: update.invoke({"monitor_id": monitor_id, field: []}),
                ACCOUNT_DETAIL[source],
            )

        def kept(result: Any) -> Optional[str]:
            expected = {**SOCIAL_CLEANED, "meta_ads": {"count": 30}}
            got = _social_options(result)
            return None if got == expected else f"options={got!r}"

        smoke.run(
            "monitor_update_tool (meta_ads_count keeps every account)",
            lambda: update.invoke({"monitor_id": monitor_id, "meta_ads_count": 30}),
            kept,
        )
        smoke.run(
            "monitor_items_tool (source meta_ads)",
            lambda: items.invoke({"monitor_id": monitor_id, "source": "meta_ads"}),
            _has_list("items"),
        )
    finally:
        for monitor_id in created:
            smoke.run(
                "monitor_control_tool delete (social sources)",
                lambda monitor_id=monitor_id: control.invoke(
                    {"monitor_id": monitor_id, "action": "delete"}
                ),
                _expect(monitor_id=monitor_id, deleted=True),
            )


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
    parser.add_argument(
        "--with-accounts",
        action="store_true",
        help=(
            "also check TikTok and watched accounts (desearch-public-api#635), then "
            "Instagram, Facebook, Meta ads and Threads accounts (#644)"
        ),
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
    if args.with_accounts:
        check_watched_accounts(smoke, args.base_url)
        check_social_sources(smoke, args.base_url)
    print(
        f"{smoke.passed} passed, {smoke.failed} failed, {smoke.skipped} skipped "
        f"on {args.base_url}"
    )
    return 1 if smoke.failed else 0


if __name__ == "__main__":
    sys.exit(main())
