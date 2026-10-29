"""Company Search tools, tested against a local HTTP server (never the live API)."""

import asyncio
import threading
from pathlib import Path

import pytest
from aiohttp import web
from pydantic import ValidationError

from langchain_desearch.companies import compact_company, company_next_step
from langchain_desearch.search_tools import company_tools, search_tools
from langchain_desearch.tools import (
    CompanyEnrichTool,
    CompanyEnrichToolInput,
    CompanySearchTool,
    CompanySearchToolInput,
    CompanyTaskControlTool,
    CompanyTasksTool,
)

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0c9f3a52-7d1e-4f0a-9a3b-2b8e6f1d4c77"
TASK_PATH = f"/desearch/companies/tasks/{TASK_ID}"


def _task(status, result_count=0, **extra):
    return {
        "task_id": TASK_ID,
        "search_type": "company_search",
        "platform": "company",
        "status": status,
        "stage": None if status == "completed" else "searching",
        "query": "Series A fintech startups in Berlin",
        "count": 25,
        "result_count": result_count,
        "progress": {"candidates_found": 40},
        "errors": [],
        "completion_reason": "count_reached" if status == "completed" else None,
        "coverage_limited": False,
        "warnings": [],
        "metadata": {},
        "reserved_cost_usd": 2.25,
        "final_cost_usd": 0.18 if status == "completed" else None,
        "created_at": "2026-10-28T10:00:00Z",
        **extra,
    }


ACME = {
    "id": "company:acme.com",
    "platform": "company",
    "handle": "acme.com",
    "domain": "acme.com",
    "url": "https://acme.com/",
    "name": "Acme",
    "description": "Payment infrastructure for marketplaces.",
    "logo_url": "https://acme.com/logo.png",
    "location": {"country": "DE", "city": "Berlin", "address": None},
    "industry": "Fintech",
    "emails": [
        {
            "value": "hello@acme.com",
            "type": "general",
            "source": "website",
            "source_url": "https://acme.com/impressum",
            "person": None,
            "verification": {"status": "valid", "checks": {"mx": "ok"}},
        },
        {"value": "", "type": "other"},
    ],
    "phones": [
        {
            "value": "+493012345678",
            "type": "main",
            "source": "website",
            "source_url": "https://acme.com/impressum",
            "raw": "+49 30 1234 5678",
        }
    ],
    "socials": [
        {
            "type": "linkedin",
            "url": "https://www.linkedin.com/company/acme",
            "handle": "acme",
            "source_url": "https://acme.com/",
        },
        {"type": "x", "url": ""},
    ],
    "pages": [{"type": "home", "url": "https://acme.com/"}],
    "people": [
        {
            "name": "Anna Schmidt",
            "title": "Co-founder & CEO",
            "roles": ["ceo", "founder"],
            "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
            "x_url": None,
            "github_url": None,
            "emails": [
                {
                    "value": "anna.schmidt@acme.com",
                    "type": "personal",
                    "source": "inferred",
                    "person": "Anna Schmidt",
                    "confidence": "high",
                    "pattern": "first.last",
                    "verification": {"status": "valid"},
                }
            ],
            "phones": [],
            "evidence": {"quote": "Anna Schmidt, CEO", "source_url": "https://acme.com/impressum"},
            "verified": True,
        },
        {"title": "No name"},
    ],
    "primary_email": {
        "value": "anna.schmidt@acme.com",
        "type": "personal",
        "source": "inferred",
        "confidence": "high",
        "verification": {"status": "valid"},
    },
    "primary_contact": {
        "name": "Anna Schmidt",
        "title": "Co-founder & CEO",
        "email": "anna.schmidt@acme.com",
        "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
        "reason": "CEO, email inferred from the company's pattern and verified",
    },
    "contact_coverage": {"email": True, "phone": True, "linkedin": True},
    "enrichment": {
        "status": "complete",
        "stages": {"profile": "done", "contacts": "done", "people": "done"},
        "errors": [],
    },
    "score": {
        "value": 86,
        "summary": "Berlin payments infrastructure startup.",
        "reasons": ["Builds payment APIs", "Based in Berlin", "Raised a Series A", "Hiring"],
        "criteria_checks": [
            {"criterion": "Based in Berlin", "verdict": "met"},
            {"criterion": "Raised a Series A", "verdict": "unknown"},
            {"criterion": "Exclude consultancies", "verdict": "not_met", "is_disqualifier": True},
        ],
        "fully_verified": False,
    },
}

BETA = {
    "domain": "beta.io",
    "name": "Beta",
    "score": None,
    "enrichment": {"status": "partial", "stages": {"profile": "done", "people": "pending"}},
}


class _FakeCompanyAPI:
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
def company_api(monkeypatch):
    api = _FakeCompanyAPI()
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
    return CompanySearchTool(base_url=api.base_url, poll_interval=0.01)


def _enrich_tool(api):
    return CompanyEnrichTool(base_url=api.base_url, poll_interval=0.01)


def _results(companies, has_more=False, cursor="c1", status="completed"):
    return {
        "task_id": TASK_ID,
        "status": status,
        "results": companies,
        "next_cursor": cursor,
        "has_more": has_more,
    }


def test_company_tool_names_groups_and_descriptions():
    assert [tool.name for tool in company_tools] == [
        "company_search_tool",
        "company_enrich_tool",
        "company_tasks_tool",
        "company_task_control_tool",
    ]
    # The billed, slow tools stay out of the default agent tool list.
    default_names = [tool.name for tool in search_tools]
    assert "company_search_tool" not in default_names
    assert "company_enrich_tool" not in default_names

    for tool in (CompanySearchTool(), CompanyEnrichTool()):
        assert "BILLED" in tool.description
        assert "task_id" in tool.description
    assert "minimum of 2" in CompanySearchTool().description
    assert "minimum of 1" in CompanyEnrichTool().description
    assert "Free" in CompanyTasksTool().description
    assert "cannot be undone" in CompanyTaskControlTool().description


def test_company_search_does_not_require_unpublished_sdk():
    for name in ("pyproject.toml", "setup.py"):
        text = (ROOT / name).read_text()
        assert "desearch-py>=1.8" not in text
        assert "desearch-py>=1.1.0,<2.0.0" in text


def test_company_search_starts_waits_and_returns_compact_companies(company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("queued", cost_usd=2.25))
    company_api.add("GET", TASK_PATH, 200, _task("running", 1, stage="enriching"))
    company_api.add("GET", TASK_PATH, 200, _task("completed", 2))
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME, BETA]))

    result = _search_tool(company_api).invoke(
        {
            "query": "Series A fintech startups in Berlin",
            "count": 25,
            "min_score": 60,
            "exclude": ["stripe.com"],
            "countries": ["DE"],
            "require": ["email", "leader", "email"],
            "roles": ["ceo", "cto"],
            "infer_emails": True,
            "verify_emails": False,
            "metadata": {"campaign": "q4"},
            "idempotency_key": "run-1",
            "timeout": 30,
        }
    )

    assert result["task_id"] == TASK_ID
    assert result["search_type"] == "company_search"
    assert result["status"] == "completed"
    assert result["done"] is True
    assert result["completion_reason"] == "count_reached"
    assert "coverage_limited" not in result
    assert result["final_cost_usd"] == 0.18
    assert "existing_task" not in result
    assert result["next_step"].startswith("Finished. These are all the companies")
    assert "1 company stayed partial" in result["next_step"]
    acme, beta = result["companies"]
    assert acme["domain"] == "acme.com"
    assert acme["score"] == 86
    assert beta == {
        "domain": "beta.io",
        "name": "Beta",
        "score": None,
        "enrichment": {"status": "partial", "stages": {"profile": "done", "people": "pending"}},
    }

    (create,) = company_api.calls("POST")
    assert create["path"] == "/desearch/companies/search"
    assert create["headers"]["Authorization"] == "test-key"
    assert create["headers"]["Idempotency-Key"] == "run-1"
    assert create["json"] == {
        "query": "Series A fintech startups in Berlin",
        "count": 25,
        "min_score": 60,
        "exclude": ["stripe.com"],
        "filters": {"countries": ["DE"], "require": ["email", "leader"]},
        "roles": ["ceo", "cto"],
        "infer_emails": True,
        "verify_emails": False,
        "metadata": {"campaign": "q4"},
    }
    assert len(company_api.calls("GET", TASK_PATH)) == 2
    (results_call,) = company_api.calls("GET", f"{TASK_PATH}/results")
    # No cursor: every company in its latest version.
    assert results_call["query"] == {"limit": "200"}
    assert all(r["headers"]["Authorization"] == "test-key" for r in company_api.requests)


def test_compact_company_keeps_contacts_leaders_and_sources():
    assert compact_company(ACME) == {
        "domain": "acme.com",
        "name": "Acme",
        "url": "https://acme.com/",
        "industry": "Fintech",
        "location": {"country": "DE", "city": "Berlin"},
        "description": "Payment infrastructure for marketplaces.",
        "score": 86,
        "why": "Berlin payments infrastructure startup.",
        "reasons": ["Builds payment APIs", "Based in Berlin", "Raised a Series A"],
        "fully_verified": False,
        "unverified_criteria": ["Raised a Series A (unknown)"],
        "primary_email": {
            "value": "anna.schmidt@acme.com",
            "type": "personal",
            "source": "inferred",
            "confidence": "high",
            "verification": "valid",
        },
        "primary_contact": {
            "name": "Anna Schmidt",
            "title": "Co-founder & CEO",
            "email": "anna.schmidt@acme.com",
            "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
            "reason": "CEO, email inferred from the company's pattern and verified",
        },
        "contact_coverage": {"email": True, "phone": True, "linkedin": True},
        "emails": [
            {
                "value": "hello@acme.com",
                "type": "general",
                "source": "website",
                "source_url": "https://acme.com/impressum",
                "verification": "valid",
            }
        ],
        "phones": [
            {
                "value": "+493012345678",
                "type": "main",
                "source_url": "https://acme.com/impressum",
            }
        ],
        "socials": [
            {
                "type": "linkedin",
                "url": "https://www.linkedin.com/company/acme",
                "source_url": "https://acme.com/",
            }
        ],
        "leaders": [
            {
                "name": "Anna Schmidt",
                "title": "Co-founder & CEO",
                "roles": ["ceo", "founder"],
                "verified": True,
                "linkedin_url": "https://www.linkedin.com/in/annaschmidt",
                "emails": [
                    {
                        "value": "anna.schmidt@acme.com",
                        "type": "personal",
                        "source": "inferred",
                        "confidence": "high",
                        "pattern": "first.last",
                        "verification": "valid",
                    }
                ],
                "evidence_url": "https://acme.com/impressum",
            }
        ],
        "enrichment": {
            "status": "complete",
            "stages": {"profile": "done", "contacts": "done", "people": "done"},
        },
    }


def test_compact_company_marks_disqualifiers_that_apply_and_truncates():
    company = compact_company(
        {
            "domain": "gamma.dev",
            "description": "x" * 400,
            "score": {
                "value": 70,
                "criteria_checks": [
                    {"criterion": "Not a consultancy (disqualifier)", "verdict": "met"},
                    {"criterion": "Sells to banks", "verdict": "not_met"},
                ],
            },
        }
    )
    assert company["unverified_criteria"] == [
        "Not a consultancy (disqualifier) (applies)",
        "Sells to banks (not met)",
    ]
    assert len(company["description"]) == 300
    assert company["description"].endswith("...")


def test_company_search_returns_companies_so_far_when_it_times_out(company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("running", 1))
    company_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([BETA], status="running")
    )

    result = _search_tool(company_api).invoke(
        {"query": "Dental clinics in Lisbon", "timeout": 0}
    )

    assert result["status"] == "running"
    assert result["done"] is False
    assert [company["domain"] for company in result["companies"]] == ["beta.io"]
    assert f"task_id='{TASK_ID}'" in result["next_step"]
    assert "free" in result["next_step"]
    assert "'partial'" in result["next_step"]
    # timeout=0 returns without polling the task.
    assert company_api.calls("GET", TASK_PATH) == []


def test_company_search_skips_results_until_any_are_delivered(company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("queued"))

    result = _search_tool(company_api).invoke({"query": "Dental clinics", "timeout": 0})

    assert result["companies"] == []
    assert company_api.calls("GET", f"{TASK_PATH}/results") == []


def test_company_search_with_task_id_waits_without_starting_a_task(company_api):
    company_api.add("GET", TASK_PATH, 200, _task("running", 1))
    company_api.add(
        "GET",
        TASK_PATH,
        200,
        _task(
            "completed",
            1,
            completion_reason="time_limit",
            coverage_limited=True,
            warnings=[
                {
                    "code": "enrichment_time_limit",
                    "message": "Some companies were not fully enriched before the search's time limit.",
                },
                {"code": "planning_fallback", "message": "The search was planned without AI."},
            ],
        ),
    )
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME]))

    result = _search_tool(company_api).invoke({"task_id": TASK_ID, "timeout": 30})

    assert company_api.calls("POST") == []
    assert result["coverage_limited"] is True
    assert [warning["code"] for warning in result["warnings"]] == [
        "enrichment_time_limit",
        "planning_fallback",
    ]
    assert result["next_step"].startswith("Finished early: a budget stopped it (time_limit)")
    assert "25 companies" in result["next_step"]


def test_company_search_reads_every_results_page(company_api):
    company_api.add("POST", "/desearch/companies/search", 202, _task("completed", 2))
    company_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([ACME], has_more=True, cursor="p2")
    )
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([BETA], cursor="p3"))

    result = _search_tool(company_api).invoke({"query": "Series A fintech startups"})

    assert [company["domain"] for company in result["companies"]] == ["acme.com", "beta.io"]
    pages = company_api.calls("GET", f"{TASK_PATH}/results")
    assert [page["query"] for page in pages] == [
        {"limit": "200"},
        {"limit": "200", "cursor": "p2"},
    ]


def test_company_search_marks_an_idempotent_replay(company_api):
    company_api.add("POST", "/desearch/companies/search", 200, _task("completed", 0))

    result = _search_tool(company_api).invoke(
        {"query": "Series A fintech startups", "idempotency_key": "same-key"}
    )

    assert result["existing_task"] is True
    assert result["companies"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "ab"},
        {"query": "Fintech startups", "count": 4},
        {"query": "Fintech startups", "count": 201},
        {"query": "Fintech startups", "min_score": 101},
        {"query": "Fintech startups", "require": ["website"]},
        {"query": "Fintech startups", "roles": ["intern"]},
        {"query": "Fintech startups", "countries": ["DE"] * 21},
        {"query": "Fintech startups", "metadata": {"Campaign": "q4"}},
        {"query": "Fintech startups", "metadata": {"campaign": "x" * 201}},
        {"query": "Fintech startups", "timeout": 1801},
    ],
)
def test_company_search_input_is_validated_before_any_call(payload):
    with pytest.raises(ValidationError):
        CompanySearchToolInput(**payload)


def test_company_search_run_refuses_bad_values_without_calling(company_api):
    with pytest.raises(ValueError, match="Unknown require: website"):
        _search_tool(company_api)._run(query="Fintech startups", require=["website"])
    assert company_api.requests == []


def test_company_enrich_sends_domains_and_returns_companies(company_api):
    enrich_task = _task(
        "completed",
        1,
        search_type="company_enrich",
        platform="company_enrich",
        completion_reason="all_domains_processed",
    )
    company_api.add("POST", "/desearch/companies/enrich", 202, {**enrich_task, "status": "queued"})
    company_api.add("GET", TASK_PATH, 200, enrich_task)
    company_api.add(
        "GET", f"{TASK_PATH}/results", 200, _results([{**ACME, "score": None}])
    )

    result = _enrich_tool(company_api).invoke(
        {
            "domains": ["acme.com", " https://www.beta.io/about ", ""],
            "roles": ["ceo"],
            "infer_emails": False,
            "idempotency_key": "enrich-1",
        }
    )

    (create,) = company_api.calls("POST")
    assert create["path"] == "/desearch/companies/enrich"
    assert create["headers"]["Idempotency-Key"] == "enrich-1"
    assert create["json"] == {
        "domains": ["acme.com", "https://www.beta.io/about"],
        "roles": ["ceo"],
        "infer_emails": False,
    }
    assert result["search_type"] == "company_enrich"
    assert result["next_step"].startswith("Finished: it read every domain.")
    (company,) = result["companies"]
    assert company["domain"] == "acme.com"
    assert company["score"] is None
    assert "why" not in company


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"domains": []},
        {"domains": ["acme.com"] * 201},
        {"domains": ["acme.com"], "roles": ["intern"]},
    ],
)
def test_company_enrich_input_is_validated_before_any_call(payload):
    with pytest.raises(ValidationError):
        CompanyEnrichToolInput(**payload)


def test_company_enrich_with_task_id_waits_without_starting_a_task(company_api):
    company_api.add("GET", TASK_PATH, 200, _task("paused", 0, search_type="company_enrich"))

    result = _enrich_tool(company_api).invoke({"task_id": TASK_ID, "timeout": 0})

    assert company_api.calls("POST") == []
    assert result["status"] == "paused"
    assert "company_task_control_tool" in result["next_step"]
    assert "action='resume'" in result["next_step"]


def test_company_tasks_sends_filters_and_returns_compact_tasks(company_api):
    company_api.add(
        "GET",
        "/desearch/companies/tasks",
        200,
        {"tasks": [_task("completed", 3, metadata={"campaign": "q4"})], "next_cursor": "n2"},
    )

    result = CompanyTasksTool(base_url=company_api.base_url).invoke(
        {
            "type": "enrich",
            "status": ["running", "paused"],
            "limit": 5,
            "cursor": "n1",
            "metadata": {"campaign": "q4"},
        }
    )

    (call,) = company_api.calls("GET", "/desearch/companies/tasks")
    assert call["headers"]["Authorization"] == "test-key"
    assert call["query"] == {
        "type": "enrich",
        "status": "running,paused",
        "limit": "5",
        "cursor": "n1",
        "metadata.campaign": "q4",
    }
    (task,) = result["tasks"]
    assert task["task_id"] == TASK_ID
    assert task["metadata"] == {"campaign": "q4"}
    assert task["done"] is True
    assert result["next_cursor"] == "n2"
    assert "cursor='n2'" in result["next_step"]


def test_company_tasks_defaults_and_async(company_api):
    company_api.add("GET", "/desearch/companies/tasks", 200, {"tasks": [], "next_cursor": None})

    result = asyncio.run(CompanyTasksTool(base_url=company_api.base_url).ainvoke({}))

    assert result == {"tasks": []}
    (call,) = company_api.calls("GET", "/desearch/companies/tasks")
    assert call["query"] == {"limit": "20"}


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
def test_company_task_control_posts_the_action(company_api, action):
    status = {"pause": "pausing", "resume": "running", "cancel": "cancelled"}[action]
    company_api.add("POST", f"{TASK_PATH}/{action}", 200, _task(status, 2))

    result = CompanyTaskControlTool(base_url=company_api.base_url).invoke(
        {"task_id": TASK_ID, "action": action}
    )

    (call,) = company_api.calls("POST")
    assert call["path"] == f"{TASK_PATH}/{action}"
    assert call["headers"]["Authorization"] == "test-key"
    assert result["status"] == status
    assert result["next_step"]


def test_company_task_control_conflict_keeps_the_api_detail(company_api):
    company_api.add(
        "POST",
        f"{TASK_PATH}/resume",
        409,
        {"detail": "Only a paused task can be resumed.", "task": _task("completed", 2)},
    )

    result = CompanyTaskControlTool(base_url=company_api.base_url).invoke(
        {"task_id": TASK_ID, "action": "resume"}
    )

    assert result == (
        "An error occurred while calling Desearch: HTTP 409: "
        "Only a paused task can be resumed. Task status: completed."
    )


def test_company_search_http_errors_are_returned_without_the_api_key(company_api):
    company_api.add(
        "POST",
        "/desearch/companies/search",
        402,
        {"detail": "Insufficient balance for this task."},
    )

    result = _search_tool(company_api).invoke({"query": "Series A fintech startups"})

    assert result == (
        "An error occurred while calling Desearch: "
        "HTTP 402: Insufficient balance for this task."
    )
    assert "test-key" not in result


def test_company_search_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        CompanySearchTool().invoke({"query": "Series A fintech startups"})


def test_company_search_sync_invoke_works_inside_a_running_loop(company_api):
    company_api.add("GET", TASK_PATH, 200, _task("completed", 1))
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME]))

    async def _call():
        return _search_tool(company_api).invoke({"task_id": TASK_ID})

    result = asyncio.run(_call())
    assert result["status"] == "completed"
    assert result["companies"][0]["domain"] == "acme.com"


def test_company_enrich_async_invoke(company_api):
    company_api.add("POST", "/desearch/companies/enrich", 202, _task("queued"))
    company_api.add("GET", TASK_PATH, 200, _task("completed", 1))
    company_api.add("GET", f"{TASK_PATH}/results", 200, _results([ACME]))

    result = asyncio.run(_enrich_tool(company_api).ainvoke({"domains": ["acme.com"]}))

    assert result["status"] == "completed"
    (create,) = company_api.calls("POST")
    assert create["json"] == {"domains": ["acme.com"]}


@pytest.mark.parametrize(
    "task, expected",
    [
        ({"status": "completed", "completion_reason": "no_candidates"}, "found no company"),
        ({"status": "completed", "completion_reason": "zero_qualified"}, "none fit"),
        ({"status": "partial"}, "Finished with partial results"),
        ({"status": "failed"}, "not charged"),
        ({"status": "cancelled"}, "rest of the reservation is returned"),
        ({"status": "expired"}, "30 days"),
        ({"status": "cancelling"}, "Cancelling."),
        ({"status": "running", "stage": "enriching"}, "Every company is delivered"),
    ],
)
def test_company_next_step(task, expected):
    step = company_next_step(task, TASK_ID, "company_search_tool", "company_task_control_tool")
    assert expected in step
