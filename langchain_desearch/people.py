"""People Search on the Desearch API (``/desearch/people/*``).

desearch-py adds People Search methods in 1.4.0, which is not on PyPI yet, so
this module calls the routes directly with aiohttp (installed with desearch-py)
and the same ``Authorization: <api key>`` header the SDK sends. Switch the
tools to the SDK methods once desearch-py 1.4.0 is published.

- ``POST /desearch/people/search/{platform}`` starts a background search and is
  billed: it reserves ``count`` profiles at the route price and settles to the
  profiles it delivers. A search that finds no one is free.
- Reading a task, its results and the profiles library is free.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

PEOPLE_API_BASE_URL = "https://api.desearch.ai"

PEOPLE_PLATFORMS = ("x", "youtube", "linkedin", "tiktok")

# Statuses after which a task never changes again.
PEOPLE_ENDED_STATUSES = frozenset(
    {"completed", "partial", "failed", "cancelled", "expired"}
)

# Channels accepted by the ``has`` filter of the profiles library.
PEOPLE_CHANNELS = ("email", "website", "x", "linkedin", "instagram", "youtube", "tiktok")

PEOPLE_PROFILE_SORTS = ("score", "recent", "audience")

# Search filters each platform accepts in the request ``filters`` object. The
# API rejects other keys with HTTP 422, so they are refused before the billed
# call is made.
PEOPLE_SEARCH_FILTERS: Dict[str, Tuple[str, ...]] = {
    "x": (
        "min_followers",
        "max_followers",
        "verified_only",
        "countries",
        "keywords",
        "exclude_keywords",
    ),
    "youtube": (
        "min_subscribers",
        "max_subscribers",
        "countries",
        "require_contact",
        "keywords",
        "exclude_keywords",
    ),
    "linkedin": (
        "min_followers",
        "max_followers",
        "countries",
        "keywords",
        "exclude_keywords",
    ),
    "tiktok": (
        "min_followers",
        "max_followers",
        "require_contact",
        "keywords",
        "exclude_keywords",
    ),
}

_REQUEST_TIMEOUT_SECONDS = 120
_RESULTS_PAGE_SIZE = 200
_MAX_RESULT_PAGES = 25


class DesearchPeopleError(Exception):
    """An HTTP error from a People Search route. The message never holds the API key."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


def _error_detail(data: Any, text: str) -> str:
    if isinstance(data, dict) and data.get("detail") is not None:
        detail = data["detail"]
        if not isinstance(detail, str):
            detail = json.dumps(detail)
        task = data.get("task")
        if isinstance(task, dict) and isinstance(task.get("status"), str):
            return f"{detail} Task status: {task['status']}."
        return detail
    return text if len(text) <= 500 else f"{text[:500]}..."


class PeopleClient:
    """Minimal async client for the People Search routes."""

    def __init__(self, api_key: str, base_url: str = PEOPLE_API_BASE_URL) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "PeopleClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _call(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
            )
        request_headers = {"Authorization": self._api_key}
        if headers:
            request_headers.update(headers)
        async with self._session.request(
            method,
            f"{self._base_url}{path}",
            params=params,
            json=body,
            headers=request_headers,
        ) as response:
            status = response.status
            text = await response.text()
        data: Any = text
        if text:
            try:
                data = json.loads(text)
            except ValueError:
                pass
        if status >= 400:
            raise DesearchPeopleError(status, _error_detail(data, text))
        return status, data

    @staticmethod
    def _task_path(task_id: str, suffix: str = "") -> str:
        return f"/desearch/people/tasks/{quote(str(task_id), safe='')}{suffix}"

    async def start_search(
        self,
        platform: str,
        body: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Start a billed search. Returns the HTTP status (202 new, 200 replayed) and the task."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self._call(
            "POST",
            f"/desearch/people/search/{quote(platform, safe='')}",
            body=body,
            headers=headers,
        )

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        _, data = await self._call("GET", self._task_path(task_id))
        return data

    async def get_results(
        self, task_id: str, cursor: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        params = _drop_none({"cursor": cursor, "limit": limit})
        _, data = await self._call(
            "GET", self._task_path(task_id, "/results"), params=params
        )
        return data

    async def list_profiles(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _, data = await self._call("GET", "/desearch/people/profiles", params=params)
        return data


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def people_search_body(
    platform: str,
    query: str,
    count: Optional[int] = None,
    min_score: Optional[int] = None,
    exclude: Optional[List[str]] = None,
    **filters: Any,
) -> Dict[str, Any]:
    """Build the request body. Raises ValueError for a filter the platform does not accept."""
    if platform not in PEOPLE_SEARCH_FILTERS:
        raise ValueError(
            f"Unknown platform {platform!r}. Use one of: {', '.join(PEOPLE_PLATFORMS)}."
        )
    allowed = PEOPLE_SEARCH_FILTERS[platform]
    set_filters = _drop_none(filters)
    rejected = [name for name in set_filters if name not in allowed]
    if rejected:
        verb = "does" if len(rejected) == 1 else "do"
        raise ValueError(
            f"{', '.join(rejected)} {verb} not apply to {platform}. "
            f"{platform} filters: {', '.join(allowed)}. Nothing was charged."
        )
    body = _drop_none(
        {"query": query, "count": count, "min_score": min_score, "exclude": exclude}
    )
    if set_filters:
        body["filters"] = set_filters
    return body


def profile_filter_params(
    platform: str,
    q: Optional[str] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    has: Optional[List[str]] = None,
    min_audience: Optional[int] = None,
    max_audience: Optional[int] = None,
    min_score: Optional[int] = None,
    location: Optional[str] = None,
    verified: Optional[bool] = None,
) -> Dict[str, Any]:
    """Query parameters for ``GET /desearch/people/profiles``."""
    params = _drop_none(
        {
            "platform": platform,
            "q": q,
            "sort": sort,
            "limit": limit,
            "cursor": cursor,
            "min_audience": min_audience,
            "max_audience": max_audience,
            "min_score": min_score,
            "location": location,
        }
    )
    if has:
        params["has"] = ",".join(dict.fromkeys(str(channel) for channel in has))
    if verified is not None:
        params["verified"] = "true" if verified else "false"
    return params


def _obj(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _compact(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Drop empty values so each person stays small in an LLM context."""
    return {
        key: value
        for key, value in fields.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def compact_person(person: Any) -> Dict[str, Any]:
    """One person as an agent needs them: who, audience, fit, why, and how to reach them."""
    p = _obj(person)
    metrics = _obj(p.get("metrics"))
    attributes = _obj(p.get("attributes"))
    score = _obj(p.get("score"))
    audience = _number(metrics.get("followers"))
    if audience is None:
        audience = _number(metrics.get("subscribers"))
    reasons = [reason for reason in _list(score.get("reasons")) if _text(reason)]
    why = _text(score.get("summary")) or (reasons[0] if reasons else None)
    evidence_url = next(
        (
            _text(item.get("url"))
            for item in map(_obj, _list(p.get("evidence")))
            if _text(item.get("url"))
        ),
        None,
    )
    found_by = p.get("found_by")
    return _compact(
        {
            "name": _text(p.get("name")),
            "handle": p.get("handle"),
            "platform": p.get("platform"),
            "url": _text(p.get("url")),
            "audience": audience,
            "location": _text(p.get("location")),
            "verified": attributes.get("verified")
            if isinstance(attributes.get("verified"), bool)
            else None,
            "headline": _text(attributes.get("headline")),
            "company": _text(attributes.get("company")),
            "score": _number(score.get("value")),
            "why": why,
            "contacts": [
                _compact({"type": item.get("type"), "value": item.get("value")})
                for item in map(_obj, _list(p.get("contacts")))
                if _text(item.get("value"))
            ],
            "links": [
                _compact({"type": item.get("type"), "url": item.get("url")})
                for item in map(_obj, _list(p.get("links")))
                if _text(item.get("url"))
            ],
            "evidence_url": evidence_url,
            "found_by": _compact(
                {
                    "task_id": found_by.get("task_id"),
                    "query": found_by.get("query"),
                    "times_found": found_by.get("times_found"),
                    "found_at": found_by.get("found_at"),
                }
            )
            if isinstance(found_by, dict)
            else None,
        }
    )


def compact_task(task: Any) -> Dict[str, Any]:
    """Task fields an agent needs: status, progress counts and cost."""
    t = _obj(task)
    status = t.get("status")
    return _compact(
        {
            "task_id": t.get("task_id"),
            "platform": t.get("platform"),
            "status": status,
            "stage": t.get("stage"),
            "done": status in PEOPLE_ENDED_STATUSES if isinstance(status, str) else None,
            "query": t.get("query"),
            "count": t.get("count"),
            "result_count": t.get("result_count"),
            "reserved_cost_usd": t.get("reserved_cost_usd"),
            "final_cost_usd": t.get("final_cost_usd"),
            "refunded_usd": t.get("refunded_usd"),
            "errors": t.get("errors"),
            "created_at": t.get("created_at"),
            "finished_at": t.get("finished_at"),
            "results_expire_at": t.get("results_expire_at"),
        }
    )


def people_next_step(status: Optional[str], task_id: Optional[str], tool_name: str) -> str:
    """One sentence on what to do next, so an agent does not infer it from the status."""
    if status == "completed":
        return "Finished. These are all the people this search found."
    if status == "partial":
        return (
            "Finished with partial results (some sub-searches failed). "
            "These are all the people it found."
        )
    if status == "failed":
        return "The search failed; see errors. A search that finds no one is not charged."
    if status == "cancelled":
        return (
            "Cancelled. You pay only for the people it delivered; "
            "the rest of the reservation is returned."
        )
    if status == "expired":
        return "The results expired (they are kept for 30 days after a search ends)."
    if status == "paused":
        return (
            f"Paused. It will not finish until it is resumed in the Desearch console "
            f"or API. Task id: {task_id}."
        )
    return (
        f"Still {status or 'running'}. The people found so far are listed. Call "
        f"{tool_name} again with task_id='{task_id}' to keep waiting; that is free "
        "and does not start or bill a new search."
    )


async def read_all_results(client: PeopleClient, task_id: str) -> List[Dict[str, Any]]:
    """Read every person a task has found so far, following the cursor."""
    people: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(_MAX_RESULT_PAGES):
        page = _obj(await client.get_results(task_id, cursor=cursor, limit=_RESULTS_PAGE_SIZE))
        results = _list(page.get("results"))
        people.extend(results)
        cursor = page.get("next_cursor")
        if not page.get("has_more") or not results or not cursor:
            break
    return people


async def search_people(
    client: PeopleClient,
    *,
    tool_name: str,
    timeout: float,
    poll_interval: float,
    platform: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a search (or pick up ``task_id``), wait up to ``timeout`` seconds, return compact people."""
    replayed = False
    if task_id:
        task = _obj(await client.get_task(task_id))
    else:
        if platform is None or body is None:
            raise ValueError("platform and body are required to start a People Search.")
        http_status, created = await client.start_search(platform, body, idempotency_key)
        task = _obj(created)
        # 200 means this Idempotency-Key already created the task; nothing new was charged.
        replayed = http_status == 200
        task_id = task.get("task_id")
        if not task_id:
            raise DesearchPeopleError(http_status, "The response did not include a task_id.")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(float(timeout), 0.0)
    while task.get("status") not in PEOPLE_ENDED_STATUSES:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval, remaining))
        task = _obj(await client.get_task(task_id))

    people = await read_all_results(client, task_id)
    output = compact_task(task)
    output.setdefault("task_id", task_id)
    if replayed:
        output["existing_task"] = True
    output["people"] = [compact_person(person) for person in people]
    output["next_step"] = people_next_step(task.get("status"), task_id, tool_name)
    return output


async def list_people_profiles(
    client: PeopleClient, *, tool_name: str, **filters: Any
) -> Dict[str, Any]:
    """One page of the profiles library, compacted."""
    platform = filters.get("platform")
    page = _obj(await client.list_profiles(profile_filter_params(**filters)))
    next_cursor = page.get("next_cursor")
    output: Dict[str, Any] = {
        "platform": platform,
        "total": page.get("total"),
        "people": [compact_person(person) for person in _list(page.get("profiles"))],
    }
    if next_cursor:
        output["next_cursor"] = next_cursor
        output["next_step"] = (
            f"More people match. Call {tool_name} again with the same filters and "
            f"cursor='{next_cursor}'."
        )
    return output
