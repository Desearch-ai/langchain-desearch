"""People Search tools, tested against a local HTTP server (never the live API)."""

import asyncio
import threading
from pathlib import Path

import pytest
from aiohttp import web
from pydantic import ValidationError

from langchain_desearch.people import compact_person
from langchain_desearch.search_tools import people_tools, search_tools
from langchain_desearch.tools import (
    PeopleProfilesTool,
    PeopleSearchTool,
    PeopleSearchToolInput,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "5b33f3ee-168b-4780-9c8e-63a626781872"
TASK_PATH = f"/desearch/people/tasks/{TASK_ID}"


def _task(status, result_count=0, **extra):
    return {
        "task_id": TASK_ID,
        "search_type": "x_profile_search",
        "platform": "x",
        "status": status,
        "stage": None if status == "completed" else "searching",
        "query": "Indie hackers building in public",
        "count": 25,
        "result_count": result_count,
        "progress": {"candidates": 40},
        "errors": [],
        "reserved_cost_usd": 1.0,
        "final_cost_usd": 0.2 if status == "completed" else None,
        "created_at": "2026-10-19T10:00:00Z",
        **extra,
    }


PERSON_X = {
    "id": "x:1",
    "platform": "x",
    "handle": "levelsio",
    "url": "https://x.com/levelsio",
    "name": "Pieter Levels",
    "avatar_url": "https://img.example/levelsio.jpg",
    "bio": "Building in public",
    "location": "Portugal",
    "metrics": {"followers": 600000, "following": 2000},
    "attributes": {"verified": True},
    "links": [
        {"type": "website", "url": "https://levels.io"},
        {"type": "website", "url": ""},
    ],
    "contacts": [{"type": "email", "value": "pieter@example.com"}, {"type": "email"}],
    "evidence": [
        {"type": "post", "text": "MRR update"},
        {"type": "post", "url": "https://x.com/levelsio/status/1", "text": "MRR $100k"},
    ],
    "score": {
        "value": 92,
        "qualified": True,
        "summary": "Shares MRR updates every month.",
        "reasons": ["Posts revenue numbers", "Indie founder"],
    },
}

PERSON_X_2 = {
    "id": "x:2",
    "platform": "x",
    "handle": "marc_louvion",
    "metrics": {"followers": 120000},
    "score": {"value": 81, "reasons": ["Shares revenue screenshots"]},
}


class _FakePeopleAPI:
    """Records each request and answers from scripted responses.

    A route keeps returning its last response, so ``add`` a sequence for
    polling (running, running, completed) and the last one sticks.
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
def people_api(monkeypatch):
    api = _FakePeopleAPI()
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


def _search_tool(api):
    return PeopleSearchTool(base_url=api.base_url, poll_interval=0.01)


def _profiles_tool(api):
    return PeopleProfilesTool(base_url=api.base_url)


def _results(people, has_more=False, cursor="c1", status="completed"):
    return {
        "task_id": TASK_ID,
        "status": status,
        "results": people,
        "next_cursor": cursor,
        "has_more": has_more,
    }


def test_people_tool_names_groups_and_descriptions():
    assert PeopleSearchTool().name == "people_search_tool"
    assert PeopleProfilesTool().name == "people_profiles_tool"
    assert [tool.name for tool in people_tools] == [
        "people_search_tool",
        "people_profiles_tool",
    ]
    # The billed, slow tool stays out of the default agent tool list.
    assert "people_search_tool" not in [tool.name for tool in search_tools]

    search_description = PeopleSearchTool().description
    assert "BILLED" in search_description
    assert "1-5 minutes" in search_description
    assert "task_id" in search_description
    assert "Free" in PeopleProfilesTool().description


def test_people_search_shares_the_package_desearch_py_requirement():
    # People Search still calls its routes over HTTP (see people.py); the
    # package requirement is the one the other tools need.
    for name in ("pyproject.toml", "setup.py"):
        text = (ROOT / name).read_text()
        assert "desearch-py>=1.8.0,<2.0.0" in text


def test_people_search_starts_waits_and_returns_compact_people(people_api):
    people_api.add("POST", "/desearch/people/search/x", 202, _task("queued", cost_usd=1.0))
    people_api.add("GET", TASK_PATH, 200, _task("running", 1))
    people_api.add("GET", TASK_PATH, 200, _task("completed", 2))
    people_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([PERSON_X, PERSON_X_2])
    )

    result = _search_tool(people_api).invoke(
        {
            "query": "Indie hackers building in public",
            "platform": "x",
            "count": 25,
            "min_score": 60,
            "exclude": ["someone"],
            "min_followers": 1000,
            "verified_only": True,
            "keywords": ["MRR"],
            "idempotency_key": "run-1",
            "timeout": 30,
        }
    )

    assert result["task_id"] == TASK_ID
    assert result["status"] == "completed"
    assert result["done"] is True
    assert result["final_cost_usd"] == 0.2
    assert "existing_task" not in result
    assert result["next_step"].startswith("Finished.")
    assert result["people"] == [
        {
            "name": "Pieter Levels",
            "handle": "levelsio",
            "platform": "x",
            "url": "https://x.com/levelsio",
            "audience": 600000,
            "location": "Portugal",
            "verified": True,
            "score": 92,
            "why": "Shares MRR updates every month.",
            "contacts": [{"type": "email", "value": "pieter@example.com"}],
            "links": [{"type": "website", "url": "https://levels.io"}],
            "evidence_url": "https://x.com/levelsio/status/1",
        },
        {
            "handle": "marc_louvion",
            "platform": "x",
            "audience": 120000,
            "score": 81,
            "why": "Shares revenue screenshots",
        },
    ]

    (create,) = people_api.calls("POST")
    assert create["path"] == "/desearch/people/search/x"
    assert create["headers"]["Authorization"] == "test-key"
    assert create["headers"]["Idempotency-Key"] == "run-1"
    assert create["json"] == {
        "query": "Indie hackers building in public",
        "count": 25,
        "min_score": 60,
        "exclude": ["someone"],
        "filters": {"min_followers": 1000, "verified_only": True, "keywords": ["MRR"]},
    }
    assert len(people_api.calls("GET", TASK_PATH)) == 2
    (results_call,) = people_api.calls("GET", f"{TASK_PATH}/results")
    assert results_call["query"] == {"limit": "200"}
    assert all(r["headers"]["Authorization"] == "test-key" for r in people_api.requests)


def test_people_search_returns_people_so_far_when_it_times_out(people_api):
    people_api.add("POST", "/desearch/people/search/linkedin", 202, _task("queued"))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X], status="running"))

    result = _search_tool(people_api).invoke(
        {"query": "Heads of sales at fintech startups", "platform": "linkedin", "timeout": 0}
    )

    assert result["status"] == "queued"
    assert result["done"] is False
    assert [person["handle"] for person in result["people"]] == ["levelsio"]
    assert f"task_id='{TASK_ID}'" in result["next_step"]
    assert "free" in result["next_step"]
    # timeout=0 returns without polling the task.
    assert people_api.calls("GET", TASK_PATH) == []


def test_people_search_with_task_id_waits_without_starting_a_search(people_api):
    people_api.add("GET", TASK_PATH, 200, _task("running", 1))
    people_api.add("GET", TASK_PATH, 200, _task("partial", 1, errors=["x timeout"]))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X_2]))

    result = _search_tool(people_api).invoke({"task_id": TASK_ID, "timeout": 30})

    assert people_api.calls("POST") == []
    assert result["status"] == "partial"
    assert result["errors"] == ["x timeout"]
    assert result["next_step"].startswith("Finished with partial results")
    assert [person["handle"] for person in result["people"]] == ["marc_louvion"]


def test_people_search_reads_every_results_page(people_api):
    people_api.add("POST", "/desearch/people/search/x", 202, _task("completed", 2))
    people_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([PERSON_X], has_more=True, cursor="p2")
    )
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X_2], cursor="p3"))

    result = _search_tool(people_api).invoke(
        {"query": "Indie hackers building in public", "platform": "x"}
    )

    assert [person["handle"] for person in result["people"]] == [
        "levelsio",
        "marc_louvion",
    ]
    pages = people_api.calls("GET", f"{TASK_PATH}/results")
    assert [page["query"] for page in pages] == [
        {"limit": "200"},
        {"limit": "200", "cursor": "p2"},
    ]


def test_people_search_marks_an_idempotent_replay(people_api):
    people_api.add("POST", "/desearch/people/search/x", 200, _task("completed", 0))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([]))

    result = _search_tool(people_api).invoke(
        {"query": "Indie hackers", "platform": "x", "idempotency_key": "same-key"}
    )

    assert result["existing_task"] is True
    assert result["people"] == []


def test_people_search_rejects_filters_the_platform_does_not_accept(people_api):
    with pytest.raises(ValidationError, match="verified_only does not apply to linkedin"):
        _search_tool(people_api).invoke(
            {"query": "Heads of sales", "platform": "linkedin", "verified_only": True}
        )
    with pytest.raises(ValueError, match="min_followers does not apply to youtube"):
        _search_tool(people_api)._run(
            query="Cooking creators", platform="youtube", min_followers=10
        )
    assert people_api.requests == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "Indie hackers"},
        {"platform": "x"},
        {"query": "ab", "platform": "x"},
        {"query": "Indie hackers", "platform": "x", "count": 4},
        {"query": "Indie hackers", "platform": "x", "count": 201},
        {"query": "Indie hackers", "platform": "facebook"},
    ],
)
def test_people_search_input_is_validated_before_any_call(payload):
    with pytest.raises(ValidationError):
        PeopleSearchToolInput(**payload)


def test_people_search_http_errors_are_returned_without_the_api_key(people_api):
    people_api.add(
        "POST",
        "/desearch/people/search/tiktok",
        402,
        {"detail": "Insufficient balance for this search."},
    )

    result = _search_tool(people_api).invoke(
        {"query": "Cooking creators who share recipes", "platform": "tiktok"}
    )

    assert result == (
        "An error occurred while calling Desearch: "
        "HTTP 402: Insufficient balance for this search."
    )
    assert "test-key" not in result


def test_people_search_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        PeopleSearchTool().invoke({"query": "Indie hackers", "platform": "x"})


def test_people_search_sync_invoke_works_inside_a_running_loop(people_api):
    people_api.add("GET", TASK_PATH, 200, _task("completed", 1))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X]))

    async def _call():
        return _search_tool(people_api).invoke({"task_id": TASK_ID})

    result = asyncio.run(_call())
    assert result["status"] == "completed"
    assert result["people"][0]["handle"] == "levelsio"


def test_people_search_async_invoke(people_api):
    people_api.add("POST", "/desearch/people/search/youtube", 202, _task("queued"))
    people_api.add("GET", TASK_PATH, 200, _task("completed", 1))
    people_api.add("GET", f"{TASK_PATH}/results", 200, _results([PERSON_X]))

    result = asyncio.run(
        _search_tool(people_api).ainvoke(
            {
                "query": "Cooking creators who share recipes",
                "platform": "youtube",
                "min_subscribers": 10000,
                "require_contact": True,
            }
        )
    )

    assert result["status"] == "completed"
    (create,) = people_api.calls("POST")
    assert create["json"]["filters"] == {"min_subscribers": 10000, "require_contact": True}


def test_people_profiles_sends_filters_and_returns_compact_people(people_api):
    profile = {
        **PERSON_X,
        "platform": "tiktok",
        "found_by": {
            "task_id": TASK_ID,
            "query": "Cooking creators",
            "found_at": "2026-10-19T10:05:00Z",
            "times_found": 2,
        },
    }
    people_api.add(
        "GET",
        "/desearch/people/profiles",
        200,
        {"profiles": [profile], "total": 31, "next_cursor": "next-page"},
    )

    result = _profiles_tool(people_api).invoke(
        {
            "platform": "tiktok",
            "q": "recipes",
            "sort": "audience",
            "limit": 10,
            "has": ["email", "instagram", "email"],
            "min_audience": 1000,
            "max_audience": 500000,
            "min_score": 70,
            "location": "united states",
            "verified": True,
        }
    )

    (call,) = people_api.calls("GET", "/desearch/people/profiles")
    assert call["headers"]["Authorization"] == "test-key"
    assert call["query"] == {
        "platform": "tiktok",
        "q": "recipes",
        "sort": "audience",
        "limit": "10",
        "has": "email,instagram",
        "min_audience": "1000",
        "max_audience": "500000",
        "min_score": "70",
        "location": "united states",
        "verified": "true",
    }
    assert result["platform"] == "tiktok"
    assert result["total"] == 31
    assert result["next_cursor"] == "next-page"
    assert "cursor='next-page'" in result["next_step"]
    (person,) = result["people"]
    assert person["handle"] == "levelsio"
    assert person["found_by"] == {
        "task_id": TASK_ID,
        "query": "Cooking creators",
        "times_found": 2,
        "found_at": "2026-10-19T10:05:00Z",
    }


def test_people_profiles_defaults_and_async(people_api):
    people_api.add(
        "GET",
        "/desearch/people/profiles",
        200,
        {"profiles": [], "total": 0, "next_cursor": None},
    )

    result = asyncio.run(_profiles_tool(people_api).ainvoke({"platform": "x"}))

    assert result == {"platform": "x", "total": 0, "people": []}
    (call,) = people_api.calls("GET", "/desearch/people/profiles")
    assert call["query"] == {"platform": "x", "sort": "score", "limit": "25"}


def test_people_profiles_errors_are_returned_as_messages(people_api):
    people_api.add(
        "GET",
        "/desearch/people/profiles",
        422,
        {"detail": [{"loc": ["query", "platform"], "msg": "bad platform"}]},
    )

    result = _profiles_tool(people_api).invoke({"platform": "x"})

    assert result.startswith("An error occurred while calling Desearch: HTTP 422: ")
    assert "bad platform" in result


def test_compact_person_uses_subscribers_and_linkedin_fields():
    youtube = compact_person(
        {
            "platform": "youtube",
            "handle": "@chef",
            "metrics": {"subscribers": 54000},
            "score": {"value": 77.5, "summary": "Posts weekly recipes."},
        }
    )
    assert youtube["audience"] == 54000
    assert youtube["score"] == 77.5

    linkedin = compact_person(
        {
            "platform": "linkedin",
            "handle": "jane-doe",
            "metrics": {"followers": 0},
            "attributes": {"headline": "VP Sales", "company": "Acme", "verified": "yes"},
        }
    )
    assert linkedin["audience"] == 0
    assert linkedin["headline"] == "VP Sales"
    assert linkedin["company"] == "Acme"
    assert "verified" not in linkedin
