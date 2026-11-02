"""Company Search and Company Enrich on the Desearch API (``/desearch/companies/*``).

desearch-py adds the Company Search methods in 1.8.0, which is not on PyPI yet,
so this module calls the routes directly with aiohttp, the same
``Authorization: <api key>`` header and the same ``HTTP <status>: <detail>``
errors as People Search (people.py). Switch the tools to the SDK methods once
desearch-py 1.8.0 is published.

- ``POST /desearch/companies/search`` finds companies from a description and
  ``POST /desearch/companies/enrich`` reads the domains you send. Both start a
  background task and are billed: they reserve companies at the route price
  and settle to the companies they deliver. A task that delivers none is free.
- Reading a task, its results and the task list, and pausing, resuming or
  cancelling a task, is free.

A company is delivered once the stages its request needs have run
(``enrichment.status`` ``partial``); later stages (contacts, leaders, inferred
emails, email checks) fill it in place until it is ``complete``. Reading the
results without a cursor returns every company in its latest version.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from langchain_desearch.people import (
    PEOPLE_API_BASE_URL,
    PEOPLE_ENDED_STATUSES,
    DesearchPeopleError,
    PeopleClient,
    _compact,
    _drop_none,
    _list,
    _number,
    _obj,
    _text,
)

COMPANY_API_BASE_URL = PEOPLE_API_BASE_URL

# Company tasks share People Search's statuses.
COMPANY_ENDED_STATUSES = PEOPLE_ENDED_STATUSES
COMPANY_TASK_STATUSES = (
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "completed",
    "partial",
    "failed",
    "cancelled",
    "expired",
)

# Leaders a task looks for (``roles``). The API default is ceo, founder, cto, cfo.
COMPANY_ROLES = (
    "ceo",
    "founder",
    "cto",
    "cfo",
    "coo",
    "cmo",
    "president",
    "owner",
    "managing_director",
)

# ``filters.require``: a company without all of these is not delivered or billed.
COMPANY_REQUIREMENTS = ("email", "phone", "linkedin", "leader", "leader_email")

# ``type`` on ``GET /desearch/companies/tasks``.
COMPANY_TASK_TYPES = ("search", "enrich")

COMPANY_CONTROL_ACTIONS = ("pause", "resume", "cancel")

COMPANY_DOMAINS_MAX = 200
COMPANY_EXCLUDE_MAX = 1000
COMPANY_COUNTRIES_MAX = 20

# ``metadata`` labels: the API rejects anything else with HTTP 422.
METADATA_MAX_KEYS = 10
METADATA_MAX_FILTERS = 3
METADATA_VALUE_MAX = 200
_METADATA_KEY_RE = re.compile(r"[a-z0-9_.-]{1,40}")

_RESULTS_PAGE_SIZE = 200
_MAX_RESULT_PAGES = 5
_MAX_DESCRIPTION_CHARS = 300
_MAX_REASONS = 3
_MAX_UNVERIFIED = 5
_MAX_CRITERION_CHARS = 120
_DISQUALIFIER_CRITERION = re.compile(
    r"\b(disqualif(?:y|ier|iers)|exclude|excluding|not a fit|not qualified)\b", re.I
)


class CompanyClient:
    """Minimal async client for the Company Search routes."""

    def __init__(self, api_key: str, base_url: str = COMPANY_API_BASE_URL) -> None:
        # Same transport, auth header and error text as People Search.
        self._http = PeopleClient(api_key, base_url)

    async def __aenter__(self) -> "CompanyClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.close()

    @staticmethod
    def _task_path(task_id: str, suffix: str = "") -> str:
        return f"/desearch/companies/tasks/{quote(str(task_id), safe='')}{suffix}"

    async def start_task(
        self,
        kind: str,
        body: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Start a billed ``search`` or ``enrich`` task. Returns the HTTP status (202 new, 200 replayed) and the task."""
        if kind not in COMPANY_TASK_TYPES:
            raise ValueError(f"kind must be one of: {', '.join(COMPANY_TASK_TYPES)}.")
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._http._call(
            "POST", f"/desearch/companies/{kind}", body=body, headers=headers
        )

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        _, data = await self._http._call("GET", self._task_path(task_id))
        return data

    async def get_results(
        self, task_id: str, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        params = _drop_none({"cursor": cursor, "limit": limit})
        _, data = await self._http._call(
            "GET", self._task_path(task_id, "/results"), params=params
        )
        return data

    async def control_task(self, task_id: str, action: str) -> Dict[str, Any]:
        if action not in COMPANY_CONTROL_ACTIONS:
            raise ValueError(f"action must be one of: {', '.join(COMPANY_CONTROL_ACTIONS)}.")
        _, data = await self._http._call("POST", self._task_path(task_id, f"/{action}"))
        return data

    async def list_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data = await self._http._call("GET", "/desearch/companies/tasks", params=params)
        return data


def _check_choices(name: str, values: Optional[List[str]], allowed: Tuple[str, ...]) -> None:
    unknown = [value for value in values or [] if value not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {name}: {', '.join(map(str, unknown))}. Use: {', '.join(allowed)}. "
            "Nothing was charged."
        )


def check_metadata(metadata: Optional[Dict[str, str]]) -> None:
    """The API's metadata rule, checked before the billed call."""
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object of string labels. Nothing was charged.")
    if len(metadata) > METADATA_MAX_KEYS:
        raise ValueError(
            f"metadata can have at most {METADATA_MAX_KEYS} keys. Nothing was charged."
        )
    for key, value in metadata.items():
        if not isinstance(key, str) or not _METADATA_KEY_RE.fullmatch(key):
            raise ValueError(
                f"metadata key {str(key)[:50]!r} is not allowed: keys are 1-40 characters "
                "of a-z, 0-9, '_', '.' or '-'. Nothing was charged."
            )
        if not isinstance(value, str) or len(value) > METADATA_VALUE_MAX:
            raise ValueError(
                f"metadata value of {key!r} must be a string of at most "
                f"{METADATA_VALUE_MAX} characters. Nothing was charged."
            )


def _task_options(
    body: Dict[str, Any],
    roles: Optional[List[str]],
    infer_emails: Optional[bool],
    verify_emails: Optional[bool],
    metadata: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    _check_choices("roles", roles, COMPANY_ROLES)
    check_metadata(metadata)
    body.update(
        _drop_none(
            {
                "roles": list(dict.fromkeys(roles)) if roles else None,
                "infer_emails": infer_emails,
                "verify_emails": verify_emails,
            }
        )
    )
    if metadata:
        body["metadata"] = metadata
    return body


def company_search_body(
    query: str,
    count: Optional[int] = None,
    min_score: Optional[int] = None,
    exclude: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    require: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
    infer_emails: Optional[bool] = None,
    verify_emails: Optional[bool] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Body of ``POST /desearch/companies/search``. Raises ValueError for a value the API would reject."""
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 1000:
        raise ValueError("query must be 3-1000 characters. Nothing was charged.")
    if count is not None and not 5 <= count <= 200:
        raise ValueError("count must be between 5 and 200. Nothing was charged.")
    if min_score is not None and not 0 <= min_score <= 100:
        raise ValueError("min_score must be between 0 and 100. Nothing was charged.")
    if exclude is not None and len(exclude) > COMPANY_EXCLUDE_MAX:
        raise ValueError(
            f"exclude can list at most {COMPANY_EXCLUDE_MAX} domains. Nothing was charged."
        )
    if countries is not None and len(countries) > COMPANY_COUNTRIES_MAX:
        raise ValueError(
            f"countries can list at most {COMPANY_COUNTRIES_MAX}. Nothing was charged."
        )
    _check_choices("require", require, COMPANY_REQUIREMENTS)
    body = _drop_none(
        {"query": query, "count": count, "min_score": min_score, "exclude": exclude}
    )
    filters = _drop_none(
        {
            "countries": countries or None,
            "require": list(dict.fromkeys(require)) if require else None,
        }
    )
    if filters:
        body["filters"] = filters
    return _task_options(body, roles, infer_emails, verify_emails, metadata)


def company_enrich_body(
    domains: List[str],
    roles: Optional[List[str]] = None,
    infer_emails: Optional[bool] = None,
    verify_emails: Optional[bool] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Body of ``POST /desearch/companies/enrich``. Raises ValueError for a value the API would reject."""
    if isinstance(domains, str):
        domains = [domains]
    cleaned = [domain.strip() for domain in domains or [] if isinstance(domain, str)]
    cleaned = [domain for domain in cleaned if domain]
    if not 1 <= len(cleaned) <= COMPANY_DOMAINS_MAX:
        raise ValueError(
            f"domains must list 1-{COMPANY_DOMAINS_MAX} domains or URLs. Nothing was charged."
        )
    return _task_options({"domains": cleaned}, roles, infer_emails, verify_emails, metadata)


def company_task_params(
    type: Optional[str] = None,
    status: Optional[List[str]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Query parameters for ``GET /desearch/companies/tasks``."""
    if type is not None and type not in COMPANY_TASK_TYPES:
        raise ValueError(f"type must be one of: {', '.join(COMPANY_TASK_TYPES)}.")
    if isinstance(status, str):
        status = status.split(",")
    statuses = list(dict.fromkeys(s.strip() for s in status or [] if s and s.strip()))
    unknown = [s for s in statuses if s not in COMPANY_TASK_STATUSES]
    if unknown:
        raise ValueError(
            f"Unknown status: {', '.join(unknown)}. Use: {', '.join(COMPANY_TASK_STATUSES)}."
        )
    if metadata and len(metadata) > METADATA_MAX_FILTERS:
        raise ValueError(f"Filter by at most {METADATA_MAX_FILTERS} metadata labels.")
    params = _drop_none(
        {
            "type": type,
            "status": ",".join(statuses) or None,
            "limit": limit,
            "cursor": cursor,
        }
    )
    # The API reads metadata filters as metadata.<key>=<value>, one parameter per pair.
    for key, value in (metadata or {}).items():
        params[f"metadata.{key}"] = value
    return params


def _truncated(value: Any, limit: int) -> Optional[str]:
    text = _text(value)
    if text and len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _strings(value: Any) -> List[str]:
    return [item for item in _list(value) if _text(item)]


def _compact_email(value: Any, with_person: bool = True) -> Dict[str, Any]:
    e = _obj(value)
    return _compact(
        {
            "value": _text(e.get("value")),
            "type": e.get("type"),
            "source": e.get("source"),
            "source_url": _text(e.get("source_url")),
            "person": _text(e.get("person")) if with_person else None,
            "confidence": e.get("confidence"),
            "pattern": _text(e.get("pattern")),
            # valid, invalid, risky or unknown; missing until the email checks ran.
            "verification": _text(_obj(e.get("verification")).get("status")),
        }
    )


def _compact_emails(value: Any, with_person: bool = True) -> List[Dict[str, Any]]:
    emails = [_compact_email(email, with_person) for email in _list(value)]
    return [email for email in emails if email.get("value")]


def _compact_phones(value: Any) -> List[Dict[str, Any]]:
    return [
        _compact(
            {
                "value": phone.get("value"),
                "type": phone.get("type"),
                "source_url": _text(phone.get("source_url")),
            }
        )
        for phone in map(_obj, _list(value))
        if _text(phone.get("value"))
    ]


def _compact_socials(value: Any) -> List[Dict[str, Any]]:
    return [
        _compact(
            {
                "type": social.get("type"),
                "url": social.get("url"),
                "source_url": _text(social.get("source_url")),
            }
        )
        for social in map(_obj, _list(value))
        if _text(social.get("url"))
    ]


def _compact_leader(value: Any) -> Dict[str, Any]:
    p = _obj(value)
    return _compact(
        {
            "name": _text(p.get("name")),
            "title": _text(p.get("title")),
            "roles": _strings(p.get("roles")),
            "verified": p.get("verified") if isinstance(p.get("verified"), bool) else None,
            "linkedin_url": _text(p.get("linkedin_url")),
            "x_url": _text(p.get("x_url")),
            "github_url": _text(p.get("github_url")),
            "emails": _compact_emails(p.get("emails"), with_person=False),
            "phones": _compact_phones(p.get("phones")),
            "evidence_url": _text(_obj(p.get("evidence")).get("source_url")),
        }
    )


def _unverified_criteria(score: Dict[str, Any]) -> List[str]:
    """Criteria the company's own pages do not confirm, as "<criterion> (not met|unknown)",
    plus disqualifiers that apply, as "<criterion> (applies)"."""
    unverified: List[str] = []
    for check in map(_obj, _list(score.get("criteria_checks"))):
        criterion = _text(check.get("criterion"))
        if not criterion:
            continue
        met = check.get("verdict") == "met"
        disqualifier = check.get("is_disqualifier")
        if not isinstance(disqualifier, bool):
            disqualifier = bool(_DISQUALIFIER_CRITERION.search(criterion))
        if met != disqualifier:
            continue
        if disqualifier:
            label = "applies"
        else:
            label = "not met" if check.get("verdict") == "not_met" else "unknown"
        unverified.append(f"{_truncated(criterion, _MAX_CRITERION_CHARS)} ({label})")
    if len(unverified) > _MAX_UNVERIFIED:
        extra = len(unverified) - _MAX_UNVERIFIED
        return [*unverified[:_MAX_UNVERIFIED], f"and {extra} more"]
    return unverified


def compact_company(company: Any) -> Dict[str, Any]:
    """One company as an agent needs it: what it is, how well it fits, and how to reach it and its leaders."""
    c = _obj(company)
    location = _obj(c.get("location"))
    score = _obj(c.get("score"))
    contact = _obj(c.get("primary_contact"))
    enrichment = _obj(c.get("enrichment"))
    primary_email = _compact_email(c.get("primary_email"))
    output = _compact(
        {
            "domain": _text(c.get("domain")) or _text(c.get("handle")),
            "name": _text(c.get("name")),
            "url": _text(c.get("url")),
            "industry": _text(c.get("industry")),
            "location": _compact(
                {
                    "country": _text(location.get("country")),
                    "city": _text(location.get("city")),
                    "address": _text(location.get("address")),
                }
            ),
            "description": _truncated(c.get("description"), _MAX_DESCRIPTION_CHARS),
        }
    )
    # Always present: a number for Company Search, None for Company Enrich (no fit score).
    output["score"] = _number(score.get("value"))
    output.update(
        _compact(
            {
                "why": _text(score.get("summary")),
                "reasons": _strings(score.get("reasons"))[:_MAX_REASONS],
                "fully_verified": score.get("fully_verified")
                if isinstance(score.get("fully_verified"), bool)
                else None,
                "unverified_criteria": _unverified_criteria(score),
                "primary_email": primary_email if primary_email.get("value") else None,
                "primary_contact": _compact(
                    {
                        "name": _text(contact.get("name")),
                        "title": _text(contact.get("title")),
                        "email": _text(contact.get("email")),
                        "linkedin_url": _text(contact.get("linkedin_url")),
                        "reason": _text(contact.get("reason")),
                    }
                ),
                "contact_coverage": _compact(_obj(c.get("contact_coverage"))),
                "emails": _compact_emails(c.get("emails")),
                "phones": _compact_phones(c.get("phones")),
                "socials": _compact_socials(c.get("socials")),
                "leaders": [
                    leader
                    for leader in map(_compact_leader, _list(c.get("people")))
                    if leader.get("name")
                ],
                "enrichment": _compact(
                    {
                        "status": _text(enrichment.get("status")),
                        "stages": _compact(_obj(enrichment.get("stages"))),
                        "errors": [
                            error
                            for error in (
                                _compact(
                                    {
                                        "stage": _text(item.get("stage")),
                                        "message": _text(item.get("message")),
                                    }
                                )
                                for item in map(_obj, _list(enrichment.get("errors")))
                            )
                            if error
                        ],
                    }
                ),
            }
        )
    )
    return output


def compact_company_task(task: Any) -> Dict[str, Any]:
    """Task fields an agent needs: status, why it stopped, warnings and cost."""
    t = _obj(task)
    status = t.get("status")
    return _compact(
        {
            "task_id": t.get("task_id"),
            "search_type": t.get("search_type"),
            "status": status,
            "stage": t.get("stage"),
            "done": status in COMPANY_ENDED_STATUSES if isinstance(status, str) else None,
            "query": t.get("query"),
            "metadata": t.get("metadata"),
            "count": t.get("count"),
            "result_count": t.get("result_count"),
            "completion_reason": _text(t.get("completion_reason")),
            "coverage_limited": True if t.get("coverage_limited") is True else None,
            "warnings": [
                warning
                for warning in (
                    _compact({"code": _text(w.get("code")), "message": _text(w.get("message"))})
                    for w in map(_obj, _list(t.get("warnings")))
                )
                if warning
            ],
            "reserved_cost_usd": t.get("reserved_cost_usd"),
            "final_cost_usd": t.get("final_cost_usd"),
            "refunded_usd": t.get("refunded_usd"),
            "errors": t.get("errors"),
            "created_at": t.get("created_at"),
            "finished_at": t.get("finished_at"),
            "results_expire_at": t.get("results_expire_at"),
        }
    )


def company_next_step(
    task: Dict[str, Any],
    task_id: Optional[str],
    tool_name: str,
    control_tool_name: str,
    partial: int = 0,
) -> str:
    """One or two sentences on what to do next, so an agent does not infer it from the status."""
    status = task.get("status")
    keep_waiting = (
        f"Call {tool_name} again with task_id='{task_id}' to keep waiting; that is free "
        "and does not start or bill a new task."
    )

    def ended(first: str) -> str:
        if partial:
            who = "1 company" if partial == 1 else f"{partial} companies"
            return (
                f"{first} {who} stayed partial: the task ended while some of "
                f"{'its' if partial == 1 else 'their'} stages were still pending."
            )
        return first

    if status == "completed":
        reason = task.get("completion_reason")
        if task.get("coverage_limited") is True:
            count = task.get("count")
            wanted = f"{count} companies" if isinstance(count, int) else "the count asked for"
            return ended(
                f"Finished early: a budget stopped it ({reason}) before it delivered "
                f"{wanted}, so more companies may match. These are all the companies it delivered."
            )
        if reason == "no_candidates":
            return "Finished: its searches found no company to check. Try a broader description."
        if reason == "zero_qualified":
            return (
                "Finished: it checked the companies its searches found and none fit. Try a "
                "broader description, a lower min_score or fewer require values."
            )
        if reason == "all_domains_processed":
            return ended("Finished: it read every domain. These are all the companies it delivered.")
        return ended("Finished. These are all the companies this task delivered.")
    if status == "partial":
        return ended(
            "Finished with partial results (some searches failed). "
            "These are all the companies it delivered."
        )
    if status == "failed":
        return "The task failed; see errors. A task that delivers no companies is not charged."
    if status == "cancelled":
        return ended(
            "Cancelled. You pay only for the companies it delivered; "
            "the rest of the reservation is returned."
        )
    if status == "expired":
        return "The results expired (they are kept for 30 days after a task ends)."
    if status in ("paused", "pausing"):
        return (
            f"{'Paused' if status == 'paused' else 'Pausing after the current batch'}. It will "
            f"not finish until it is resumed: call {control_tool_name} with task_id='{task_id}' "
            "and action='resume' (no new charge), or action='cancel' to stop it."
        )
    if status == "cancelling":
        return f"Cancelling. {keep_waiting}"
    if task.get("stage") == "enriching":
        return (
            "Every company is delivered; later stages (leaders, inferred emails, email "
            f"checks) are still filling some in. {keep_waiting}"
        )
    return (
        f"Still {status or 'running'}. The companies delivered so far are listed; companies "
        f"with enrichment status 'partial' are still being filled in. {keep_waiting}"
    )


async def read_all_companies(client: CompanyClient, task_id: str) -> List[Dict[str, Any]]:
    """Read every company a task has delivered so far, each in its latest version."""
    companies: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(_MAX_RESULT_PAGES):
        page = _obj(await client.get_results(task_id, cursor=cursor, limit=_RESULTS_PAGE_SIZE))
        results = _list(page.get("results"))
        companies.extend(results)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or not results or not cursor:
            break
    return companies


async def run_company_task(
    client: CompanyClient,
    *,
    tool_name: str,
    control_tool_name: str,
    timeout: float,
    poll_interval: float,
    kind: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a task (or pick up ``task_id``), wait up to ``timeout`` seconds, return compact companies."""
    replayed = False
    if task_id:
        task = _obj(await client.get_task(task_id))
    else:
        if kind is None or body is None:
            raise ValueError("kind and body are required to start a company task.")
        http_status, created = await client.start_task(kind, body, idempotency_key)
        task = _obj(created)
        # 200 means this Idempotency-Key already created the task; nothing new was charged.
        replayed = http_status == 200
        task_id = task.get("task_id")
        if not task_id:
            raise DesearchPeopleError(http_status, "The response did not include a task_id.")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(float(timeout), 0.0)
    while task.get("status") not in COMPANY_ENDED_STATUSES:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval, remaining))
        task = _obj(await client.get_task(task_id))

    # Expired results answer 410, and a task with no results has nothing to read.
    companies: List[Dict[str, Any]] = []
    if task.get("status") != "expired" and (task.get("result_count") or 0) > 0:
        companies = await read_all_companies(client, task_id)
    output = compact_company_task(task)
    output.setdefault("task_id", task_id)
    if replayed:
        output["existing_task"] = True
    output["companies"] = [compact_company(company) for company in companies]
    partial = sum(
        1 for company in companies if _obj(_obj(company).get("enrichment")).get("status") == "partial"
    )
    output["next_step"] = company_next_step(
        task, task_id, tool_name, control_tool_name, partial=partial
    )
    return output


async def control_company_task(
    client: CompanyClient,
    *,
    task_id: str,
    action: str,
    tool_name: str,
    control_tool_name: str,
    enrich_tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Pause, resume or cancel a task and return its updated status."""
    task = _obj(await client.control_task(task_id, action))
    if enrich_tool_name and task.get("search_type") == "company_enrich":
        tool_name = enrich_tool_name
    output = compact_company_task(task)
    output.setdefault("task_id", task_id)
    output["next_step"] = company_next_step(task, task_id, tool_name, control_tool_name)
    return output


async def list_company_tasks(
    client: CompanyClient, *, tool_name: str, **filters: Any
) -> Dict[str, Any]:
    """One page of the caller's Company Search and Company Enrich tasks, newest first."""
    page = _obj(await client.list_tasks(company_task_params(**filters)))
    next_cursor = page.get("next_cursor")
    output: Dict[str, Any] = {
        "tasks": [compact_company_task(task) for task in _list(page.get("tasks"))]
    }
    if next_cursor:
        output["next_cursor"] = next_cursor
        output["next_step"] = (
            f"More tasks match. Call {tool_name} again with the same filters and "
            f"cursor='{next_cursor}'."
        )
    return output
