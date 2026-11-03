"""Monitors tools, tested against a local HTTP server (never the live API)."""

import asyncio
import threading

import pytest
from aiohttp import web
from pydantic import ValidationError

from langchain_desearch.monitors import compact_item, run_next_step
from langchain_desearch.search_tools import monitor_tools, search_tools
from langchain_desearch.tools import (
    MonitorControlTool,
    MonitorCreateTool,
    MonitorCreateToolInput,
    MonitorEstimateTool,
    MonitorItemsTool,
    MonitorItemsToolInput,
    MonitorRunsTool,
    MonitorsTool,
    MonitorUpdateTool,
    MonitorUpdateToolInput,
)

MONITOR_ID = "6f1d0c9e-0d55-4b8e-9a3c-3c4d9b8e2a10"
RUN_ID = "0c6a1b2c-3d4e-4f50-8a9b-0c1d2e3f4a5b"
MONITOR_PATH = f"/desearch/monitors/{MONITOR_ID}"
RUN_PATH = f"{MONITOR_PATH}/runs/{RUN_ID}"


def _monitor(**extra):
    return {
        "monitor_id": MONITOR_ID,
        "name": "Bittensor subnet news",
        "query": "Bittensor subnet launches and updates",
        "criteria": ["Announces a new subnet, release or partnership"],
        "sources": ["web", "x"],
        "interval": "1d",
        "options": {"x": {"count": 20, "lang": "en", "min_likes": None}, "web": {"count": 10}},
        "min_score": 60,
        "summary": False,
        "status": "active",
        "paused_reason": None,
        "webhook": {
            "url": "https://example.com/hooks/desearch",
            "events": ["monitor.items.found"],
            "consecutive_failures": 0,
        },
        "metadata": {"team": "growth"},
        "max_run_cost_usd": 0.0098,
        "next_run_at": "2026-11-04T09:00:00Z",
        "last_run": None,
        "stats": {"runs": 0, "items": 0, "spent_usd": 0.0},
        "created_at": "2026-11-03T09:00:00Z",
        "updated_at": "2026-11-03T09:00:00Z",
        **extra,
    }


def _run(status, **extra):
    return {
        "run_id": RUN_ID,
        "monitor_id": MONITOR_ID,
        "trigger": "manual",
        "status": status,
        "fail_reason": None,
        "window_start": "2026-11-02T09:00:00Z",
        "window_end": "2026-11-03T09:00:00Z",
        "item_count": 0,
        "stats": {},
        "summary": None,
        "reserved_usd": 0.0098,
        "cost_usd": None,
        "created_at": "2026-11-03T09:00:00Z",
        "started_at": None,
        "finished_at": None,
        **extra,
    }


ITEM = {
    "item_id": "1842",
    "run_id": RUN_ID,
    "source": "x",
    "url": "https://x.com/opentensor/status/2117",
    "title": None,
    "text": "Subnet 64 is live on mainnet. " * 40,
    "author": {"name": "Opentensor", "handle": "opentensor", "url": "https://x.com/opentensor"},
    "published_at": "2026-11-03T08:12:00Z",
    "score": 91,
    "signals": {"relevant": 0.91, "new_info": 0.84},
    "found_at": "2026-11-03T09:00:41Z",
    "data": {"likes": 120, "reposts": 14, "replies": 9, "views": 5400},
}


class _FakeMonitorAPI:
    """Records each request and answers from scripted responses.

    A route keeps returning its last response, so ``add`` a sequence for
    polling (queued, running, completed) and the last one sticks.
    """

    def __init__(self):
        self.requests = []
        self.routes = {}
        self.base_url = ""

    def add(self, method, path, status, payload):
        self.routes.setdefault((method, path), []).append((status, payload))

    def calls(self, method=None, path=None):
        return [
            r
            for r in self.requests
            if (method is None or r["method"] == method)
            and (path is None or r["path"] == path)
        ]

    async def handle(self, request):
        body = await request.json() if request.can_read_body else None
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
                "headers": dict(request.headers),
                "json": body,
            }
        )
        queue = self.routes.get((request.method, request.path))
        if not queue:
            return web.json_response({"detail": "Not Found"}, status=404)
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return web.json_response(payload, status=status)


@pytest.fixture
def monitor_api(monkeypatch):
    api = _FakeMonitorAPI()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", api.handle)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    api.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DESEARCH_API_KEY", "test-key")
    try:
        yield api
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_monitor_tool_names_groups_and_descriptions():
    assert [tool.name for tool in monitor_tools] == [
        "monitor_create_tool",
        "monitor_estimate_tool",
        "monitors_tool",
        "monitor_update_tool",
        "monitor_control_tool",
        "monitor_runs_tool",
        "monitor_items_tool",
    ]
    # Runs are billed, so monitors stay out of the default agent tool list.
    default_names = {tool.name for tool in search_tools}
    assert not default_names & {tool.name for tool in monitor_tools}

    create = MonitorCreateTool().description
    assert "$0.05 per 1,000 items screened" in create
    assert "webhook_secret" in create and "monitor_estimate_tool" in create
    assert "BILLED" in MonitorControlTool().description
    assert "cannot be undone" in MonitorControlTool().description
    for tool in (MonitorEstimateTool(), MonitorsTool(), MonitorRunsTool(), MonitorItemsTool()):
        assert "Free" in tool.description


def test_monitor_create_posts_the_body_and_returns_the_secret_once(monitor_api):
    monitor_api.add(
        "POST", "/desearch/monitors", 201, _monitor(webhook_secret="whsec_abc")
    )

    result = MonitorCreateTool(base_url=monitor_api.base_url).invoke(
        {
            "query": "  Bittensor subnet   launches and updates ",
            "name": "Bittensor subnet news",
            "criteria": ["Announces a new subnet, release or partnership", " "],
            "sources": ["web", "x", "x"],
            "interval": "1d",
            "min_score": 70,
            "summary": True,
            "web_count": 15,
            "x_count": 40,
            "x_lang": "en",
            "webhook_url": "https://example.com/hooks/desearch",
            "webhook_events": ["monitor.items.found", "monitor.run.failed"],
            "metadata": {"team": "growth"},
        }
    )

    (call,) = monitor_api.calls("POST", "/desearch/monitors")
    assert call["headers"]["Authorization"] == "test-key"
    assert call["json"] == {
        "query": "Bittensor subnet launches and updates",
        "name": "Bittensor subnet news",
        "criteria": ["Announces a new subnet, release or partnership"],
        "sources": ["web", "x"],
        "interval": "1d",
        "min_score": 70,
        "summary": True,
        "metadata": {"team": "growth"},
        "options": {"web": {"count": 15}, "x": {"count": 40, "lang": "en"}},
        "webhook": {
            "url": "https://example.com/hooks/desearch",
            "events": ["monitor.items.found", "monitor.run.failed"],
        },
        "run_now": True,
    }
    assert result["monitor_id"] == MONITOR_ID
    assert result["webhook_secret"] == "whsec_abc"
    assert result["options"]["x"] == {"count": 20, "lang": "en", "min_likes": None}
    assert "last_run" not in result
    assert result["next_step"].startswith("The first run is queued now.")
    assert "returned only once" in result["next_step"]
    assert "monitor_items_tool" in result["next_step"]


def test_monitor_create_without_run_now_or_webhook(monitor_api):
    monitor_api.add("POST", "/desearch/monitors", 201, _monitor(webhook=None))

    result = MonitorCreateTool(base_url=monitor_api.base_url).invoke(
        {"query": "Bittensor subnet launches", "run_now": False}
    )

    (call,) = monitor_api.calls("POST")
    assert call["json"] == {"query": "Bittensor subnet launches", "run_now": False}
    assert "webhook" not in result
    assert "queued" not in result["next_step"]
    assert "every 1d; next run at 2026-11-04T09:00:00Z" in result["next_step"]
    assert "$0.0098" in result["next_step"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "ab"},
        {"query": "x" * 501},
        {"query": "valid query", "criteria": ["a"] * 6},
        {"query": "valid query", "criteria": ["a" * 301]},
        {"query": "valid query", "sources": []},
        {"query": "valid query", "sources": ["tiktok"]},
        {"query": "valid query", "interval": "2h"},
        {"query": "valid query", "min_score": 101},
        {"query": "valid query", "web_count": 12},
        {"query": "valid query", "x_count": 0},
        {"query": "valid query", "x_lang": "e"},
        {"query": "valid query", "x_min_likes": -1},
        {"query": "valid query", "webhook_url": "http://example.com/hook"},
        {"query": "valid query", "webhook_events": ["monitor.items.found"]},
        {"query": "valid query", "webhook_url": "https://e.com/h", "webhook_events": ["x"]},
        {"query": "valid query", "metadata": {f"k{i}": "v" for i in range(21)}},
        {"query": "valid query", "metadata": {"k" * 65: "v"}},
        {"query": "valid query", "metadata": {"k": "v" * 501}},
    ],
)
def test_monitor_create_input_is_validated_before_any_call(payload):
    with pytest.raises(ValidationError):
        MonitorCreateToolInput(**payload)


def test_monitor_create_run_refuses_bad_values_without_calling(monitor_api):
    with pytest.raises(ValidationError):
        MonitorCreateTool(base_url=monitor_api.base_url).invoke(
            {"query": "valid query", "webhook_url": "ftp://example.com"}
        )
    assert monitor_api.requests == []


def test_monitor_create_too_many_monitors_returns_the_api_detail(monitor_api):
    monitor_api.add(
        "POST",
        "/desearch/monitors",
        409,
        {"detail": "You already have 25 monitors. Delete one first."},
    )

    result = MonitorCreateTool(base_url=monitor_api.base_url).invoke({"query": "Bittensor"})

    assert result == (
        "An error occurred while calling Desearch: HTTP 409: "
        "You already have 25 monitors. Delete one first."
    )
    assert "test-key" not in result


def test_monitor_estimate_sends_sources_and_options(monitor_api):
    estimate = {
        "max_run_cost_usd": 0.0098,
        "runs_per_month": 30,
        "max_monthly_cost_usd": 0.294,
        "breakdown": {"x": 0.004, "web": 0.005, "screen": 0.0008},
    }
    monitor_api.add("POST", "/desearch/monitors/estimate", 200, estimate)

    result = MonitorEstimateTool(base_url=monitor_api.base_url).invoke(
        {"sources": ["x", "web"], "interval": "1d", "summary": False, "x_count": 50}
    )

    (call,) = monitor_api.calls("POST", "/desearch/monitors/estimate")
    assert call["json"] == {
        "sources": ["x", "web"],
        "interval": "1d",
        "summary": False,
        "options": {"x": {"count": 50}},
    }
    assert result == estimate


def test_monitors_lists_with_filters_and_a_cursor(monitor_api):
    monitor_api.add(
        "GET",
        "/desearch/monitors",
        200,
        {"monitors": [_monitor(), _monitor(status="paused", paused_reason="user")], "next_cursor": "n2"},
    )

    result = MonitorsTool(base_url=monitor_api.base_url).invoke(
        {"status": ["active", "paused"], "limit": 2, "cursor": "n1"}
    )

    (call,) = monitor_api.calls("GET", "/desearch/monitors")
    assert call["query"] == {"status": "active,paused", "limit": "2", "cursor": "n1"}
    assert [m["status"] for m in result["monitors"]] == ["active", "paused"]
    assert result["monitors"][1]["paused_reason"] == "user"
    assert result["next_cursor"] == "n2"
    assert "cursor='n2'" in result["next_step"]


def test_monitors_defaults_and_async(monitor_api):
    monitor_api.add("GET", "/desearch/monitors", 200, {"monitors": [], "next_cursor": None})

    result = asyncio.run(MonitorsTool(base_url=monitor_api.base_url).ainvoke({}))

    assert result == {"monitors": []}
    (call,) = monitor_api.calls("GET")
    assert call["query"] == {"limit": "20"}


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("insufficient_credits", "Add credits, then call monitor_control_tool"),
        ("webhook_failing", "action='test_webhook'"),
        ("user", "does not run on schedule until you call monitor_control_tool"),
    ],
)
def test_monitors_reads_one_monitor_and_explains_a_pause(monitor_api, reason, expected):
    last_run = {
        "run_id": RUN_ID,
        "status": "failed",
        "item_count": 0,
        "cost_usd": None,
        "fail_reason": "insufficient_credits",
        "finished_at": "2026-11-03T09:00:01Z",
    }
    monitor_api.add(
        "GET",
        MONITOR_PATH,
        200,
        _monitor(status="paused", paused_reason=reason, last_run=last_run),
    )

    result = MonitorsTool(base_url=monitor_api.base_url).invoke({"monitor_id": MONITOR_ID})

    assert [r["path"] for r in monitor_api.requests] == [MONITOR_PATH]
    assert result["last_run"] == {
        "run_id": RUN_ID,
        "status": "failed",
        "item_count": 0,
        "fail_reason": "insufficient_credits",
        "finished_at": "2026-11-03T09:00:01Z",
    }
    assert expected in result["next_step"]
    assert f"monitor_id='{MONITOR_ID}'" in result["next_step"]


def test_monitor_update_patches_only_the_settings_given(monitor_api):
    monitor_api.add("PATCH", MONITOR_PATH, 200, _monitor(interval="6h", criteria=[]))

    result = MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "interval": "6h", "criteria": [], "min_score": 75}
    )

    assert [r["method"] for r in monitor_api.requests] == ["PATCH"]
    assert monitor_api.requests[0]["json"] == {
        "interval": "6h",
        "criteria": [],
        "min_score": 75,
    }
    assert result["interval"] == "6h"
    assert "criteria" not in result


def test_monitor_update_merges_options_with_the_current_ones(monitor_api):
    monitor_api.add("GET", MONITOR_PATH, 200, _monitor())
    monitor_api.add("PATCH", MONITOR_PATH, 200, _monitor())

    MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "x_count": 50, "x_min_likes": 5}
    )

    (patch,) = monitor_api.calls("PATCH")
    assert patch["json"] == {
        "options": {"x": {"count": 50, "lang": "en", "min_likes": 5}, "web": {"count": 10}}
    }


def test_monitor_update_webhook_events_keep_the_current_url(monitor_api):
    monitor_api.add("GET", MONITOR_PATH, 200, _monitor())
    monitor_api.add("PATCH", MONITOR_PATH, 200, _monitor())

    MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "webhook_events": ["monitor.run.completed"]}
    )

    (patch,) = monitor_api.calls("PATCH")
    assert patch["json"] == {
        "webhook": {
            "url": "https://example.com/hooks/desearch",
            "events": ["monitor.run.completed"],
        }
    }


def test_monitor_update_new_webhook_url_keeps_the_current_events(monitor_api):
    monitor_api.add("GET", MONITOR_PATH, 200, _monitor())
    monitor_api.add(
        "PATCH", MONITOR_PATH, 200, _monitor(webhook_secret="whsec_new")
    )

    result = MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "webhook_url": "https://hooks.example.org/in"}
    )

    (patch,) = monitor_api.calls("PATCH")
    assert patch["json"] == {
        "webhook": {"url": "https://hooks.example.org/in", "events": ["monitor.items.found"]}
    }
    assert result["webhook_secret"] == "whsec_new"
    assert "returned only once" in result["next_step"]


def test_monitor_update_webhook_events_need_a_webhook(monitor_api):
    monitor_api.add("GET", MONITOR_PATH, 200, _monitor(webhook=None))

    result = MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "webhook_events": ["monitor.run.failed"]}
    )

    assert result == (
        "An error occurred while calling Desearch: "
        "This monitor has no webhook. Set webhook_url to add one."
    )
    assert monitor_api.calls("PATCH") == []


def test_monitor_update_remove_webhook_sends_null(monitor_api):
    monitor_api.add("PATCH", MONITOR_PATH, 200, _monitor(webhook=None))

    result = MonitorUpdateTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "remove_webhook": True}
    )

    assert [r["method"] for r in monitor_api.requests] == ["PATCH"]
    assert monitor_api.requests[0]["json"] == {"webhook": None}
    assert "webhook" not in result


@pytest.mark.parametrize(
    "payload",
    [
        {"monitor_id": MONITOR_ID},
        {"monitor_id": MONITOR_ID, "remove_webhook": True, "webhook_url": "https://e.com/h"},
        {"monitor_id": MONITOR_ID, "webhook_url": "http://e.com/h"},
        {"monitor_id": MONITOR_ID, "sources": []},
        {"monitor_id": MONITOR_ID, "web_count": 25},
        {"monitor_id": ""},
    ],
)
def test_monitor_update_input_is_validated_before_any_call(payload):
    with pytest.raises(ValidationError):
        MonitorUpdateToolInput(**payload)


@pytest.mark.parametrize("action, status", [("pause", "paused"), ("resume", "active")])
def test_monitor_control_pause_and_resume_patch_the_status(monitor_api, action, status):
    monitor_api.add("PATCH", MONITOR_PATH, 200, _monitor(status=status))

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": action}
    )

    (call,) = monitor_api.calls("PATCH", MONITOR_PATH)
    assert call["json"] == {"status": status}
    assert result["status"] == status
    assert result["next_step"]


def test_monitor_control_trigger_queues_a_run(monitor_api):
    monitor_api.add("POST", f"{MONITOR_PATH}/trigger", 202, _run("queued"))

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": "trigger"}
    )

    (call,) = monitor_api.calls("POST")
    assert call["path"] == f"{MONITOR_PATH}/trigger"
    assert call["json"] is None
    assert result["run_id"] == RUN_ID
    assert result["status"] == "queued"
    assert result["done"] is False
    assert "up to $0.0098" in result["next_step"]
    assert f"run_id='{RUN_ID}'" in result["next_step"]
    assert "monitor_runs_tool" in result["next_step"]


def test_monitor_control_trigger_while_running_keeps_the_api_detail(monitor_api):
    monitor_api.add(
        "POST", f"{MONITOR_PATH}/trigger", 409, {"detail": "This monitor is already running."}
    )

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": "trigger"}
    )

    assert result == (
        "An error occurred while calling Desearch: HTTP 409: This monitor is already running."
    )


def test_monitor_control_delete(monitor_api):
    monitor_api.add("DELETE", MONITOR_PATH, 200, {"monitor_id": MONITOR_ID, "deleted": True})

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": "delete"}
    )

    assert [r["method"] for r in monitor_api.requests] == ["DELETE"]
    assert result == {
        "monitor_id": MONITOR_ID,
        "deleted": True,
        "next_step": "Deleted: it will not run again.",
    }


@pytest.mark.parametrize(
    "answer, expected",
    [
        (
            {"delivered": True, "status_code": 204, "error": None, "duration_ms": 120},
            "Delivered: your endpoint answered HTTP 204.",
        ),
        (
            {"delivered": False, "status_code": 405, "error": None, "duration_ms": 80},
            "Not delivered (HTTP 405).",
        ),
        (
            {"delivered": False, "status_code": None, "error": "timeout", "duration_ms": 10000},
            "Not delivered (timeout).",
        ),
    ],
)
def test_monitor_control_test_webhook(monitor_api, answer, expected):
    monitor_api.add("POST", f"{MONITOR_PATH}/webhook/test", 200, answer)

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": "test_webhook"}
    )

    assert {k: result[k] for k in answer} == answer
    assert result["next_step"].startswith(expected)


def test_monitor_control_rotate_webhook_secret(monitor_api):
    monitor_api.add(
        "POST", f"{MONITOR_PATH}/webhook/rotate-secret", 200, {"webhook_secret": "whsec_new"}
    )

    result = MonitorControlTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "action": "rotate_webhook_secret"}
    )

    assert result["webhook_secret"] == "whsec_new"
    assert "returned only once" in result["next_step"]


def test_monitor_control_rejects_an_unknown_action():
    with pytest.raises(ValidationError):
        MonitorControlTool().invoke({"monitor_id": MONITOR_ID, "action": "cancel"})


def test_monitor_runs_lists_runs_with_a_cursor(monitor_api):
    monitor_api.add(
        "GET",
        f"{MONITOR_PATH}/runs",
        200,
        {"runs": [_run("completed", item_count=3, cost_usd=0.0041)], "next_cursor": "r2"},
    )

    result = MonitorRunsTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "limit": 1, "cursor": "r1"}
    )

    (call,) = monitor_api.calls("GET")
    assert call["query"] == {"limit": "1", "cursor": "r1"}
    (run,) = result["runs"]
    assert run["done"] is True
    assert run["cost_usd"] == 0.0041
    assert result["next_cursor"] == "r2"
    assert "cursor='r2'" in result["next_step"]


def test_monitor_runs_waits_for_a_run_and_returns_its_items(monitor_api):
    stats = {
        "sources": {"x": {"fetched": 20, "in_window": 18, "new": 7, "passed": 1, "error": None}},
        "screened": 7,
        "passed": 1,
        "screen": "jev",
    }
    monitor_api.add("GET", RUN_PATH, 200, _run("queued"))
    monitor_api.add("GET", RUN_PATH, 200, _run("running"))
    monitor_api.add(
        "GET",
        RUN_PATH,
        200,
        _run(
            "completed",
            item_count=1,
            cost_usd=0.0041,
            stats=stats,
            finished_at="2026-11-03T09:00:41Z",
            items=[ITEM],
        ),
    )

    result = MonitorRunsTool(base_url=monitor_api.base_url, poll_interval=0.01).invoke(
        {"monitor_id": MONITOR_ID, "run_id": RUN_ID, "timeout": 5}
    )

    assert len(monitor_api.calls("GET", RUN_PATH)) == 3
    assert result["status"] == "completed"
    assert result["stats"] == stats
    assert result["next_step"] == "Completed: it kept 1 new item and was billed $0.0041."
    (item,) = result["items"]
    assert item["item_id"] == "1842"
    assert item["author"] == {
        "name": "Opentensor",
        "handle": "opentensor",
        "url": "https://x.com/opentensor",
    }
    assert len(item["text"]) == 500 and item["text"].endswith("...")
    assert "title" not in item


def test_monitor_runs_without_timeout_reads_once(monitor_api):
    monitor_api.add("GET", RUN_PATH, 200, _run("queued"))

    result = MonitorRunsTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "run_id": RUN_ID}
    )

    assert len(monitor_api.requests) == 1
    assert result["items"] == []
    assert result["next_step"].startswith("Still queued.")
    assert f"run_id='{RUN_ID}'" in result["next_step"]


@pytest.mark.parametrize(
    "run, expected",
    [
        (
            {"status": "failed", "fail_reason": "insufficient_credits"},
            "Failed: the balance could not hold the run, so the monitor is paused until it is resumed.",
        ),
        ({"status": "failed", "fail_reason": "all_sources_failed"}, "Failed: every source failed for this window."),
        ({"status": "completed", "item_count": 0, "cost_usd": 0.002}, "Completed: it kept no new item and was billed $0.0020."),
        ({"status": "completed", "item_count": 4}, "Completed: it kept 4 new items."),
        ({"status": "cancelled"}, "Cancelled."),
    ],
)
def test_run_next_step(run, expected):
    assert run_next_step(run) == expected


def test_monitor_items_filters_by_source_and_pages(monitor_api):
    monitor_api.add(
        "GET", f"{MONITOR_PATH}/items", 200, {"items": [ITEM], "next_cursor": "1841"}
    )

    result = MonitorItemsTool(base_url=monitor_api.base_url).invoke(
        {"monitor_id": MONITOR_ID, "source": "x", "limit": 1, "cursor": "1900"}
    )

    (call,) = monitor_api.calls("GET")
    assert call["path"] == f"{MONITOR_PATH}/items"
    assert call["query"] == {"source": "x", "limit": "1", "cursor": "1900"}
    assert result["items"][0]["data"] == ITEM["data"]
    assert result["items"][0]["signals"] == {"relevant": 0.91, "new_info": 0.84}
    assert result["next_cursor"] == "1841"
    assert "cursor='1841'" in result["next_step"]


def test_monitor_items_defaults_and_async(monitor_api):
    monitor_api.add("GET", f"{MONITOR_PATH}/items", 200, {"items": [], "next_cursor": None})

    result = asyncio.run(
        MonitorItemsTool(base_url=monitor_api.base_url).ainvoke({"monitor_id": MONITOR_ID})
    )

    assert result == {"items": []}
    assert monitor_api.requests[0]["query"] == {"limit": "50"}


def test_monitor_items_rejects_an_unknown_source():
    with pytest.raises(ValidationError):
        MonitorItemsToolInput(monitor_id=MONITOR_ID, source="tiktok")


def test_compact_item_drops_empty_fields():
    item = compact_item({**ITEM, "text": "short", "author": None, "data": {"domain": None}})
    assert item["text"] == "short"
    assert "author" not in item and "data" not in item


def test_monitor_not_found_is_returned_without_the_api_key(monitor_api):
    result = MonitorsTool(base_url=monitor_api.base_url).invoke({"monitor_id": "missing"})

    assert result == "An error occurred while calling Desearch: HTTP 404: Not Found"
    assert "test-key" not in result


def test_monitor_tools_need_the_api_key(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        MonitorsTool().invoke({})


def test_monitor_sync_invoke_works_inside_a_running_loop(monitor_api):
    monitor_api.add("GET", MONITOR_PATH, 200, _monitor())

    async def _call():
        return MonitorsTool(base_url=monitor_api.base_url).invoke({"monitor_id": MONITOR_ID})

    result = asyncio.run(_call())
    assert result["monitor_id"] == MONITOR_ID


def test_monitor_create_async_invoke(monitor_api):
    monitor_api.add("POST", "/desearch/monitors", 201, _monitor())

    result = asyncio.run(
        MonitorCreateTool(base_url=monitor_api.base_url).ainvoke(
            {"query": "Bittensor subnet launches", "sources": ["reddit", "youtube", "threads"]}
        )
    )

    (call,) = monitor_api.calls("POST")
    assert call["json"]["sources"] == ["reddit", "youtube", "threads"]
    assert result["monitor_id"] == MONITOR_ID
