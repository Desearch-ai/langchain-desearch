"""Monitors on the Desearch API (``/desearch/monitors/*``).

A monitor reruns a search over the web, X, Reddit, YouTube, Threads, TikTok and
the Meta Ad Library, or watches accounts, TikTok and Instagram hashtags, and
Facebook pages, on a schedule (every 1h, 3h, 6h, 12h, 1d or 7d) and keeps only
what is new. Each run searches the window since the previous run ended (or, for
a source that watches accounts or hashtags, reads each feed's newest posts),
drops every item the monitor has already seen, screens the rest with an AI
classifier (``relevant``, ``new_info`` and one yes/no question per criterion; ``score`` is the lowest of ``relevant`` and the
criteria, 0-100) and keeps the items at ``min_score`` or above. Kept items go to
an optional signed webhook and stay readable here, newest first.

desearch-py adds the Monitors methods in 1.13.0, which is not on PyPI yet, so
this module calls the routes directly with the People Search transport
(people.py): the same ``Authorization: <api key>`` header and the same
``HTTP <status>: <detail>`` errors as Company Search (companies.py). Switch the
tools to the SDK methods once desearch-py 1.13.0 is published.

- Creating, reading, changing, pausing and deleting monitors is free, and so
  are webhook tests and secret rotation.
- Each run is billed: its searches at their list price, AI screening at $0.05
  per 1,000 items screened, and the summary's LLM usage plus 50%. A run holds
  the monitor's ``max_run_cost_usd`` on the balance and bills only what it used;
  a run with no new items still pays for its searches.
- A watched account is one feed call per run at that feed's list price: an X
  timeline (20 posts) is about $0.003, a subreddit or YouTube channel $0.00207,
  a TikTok profile or hashtag ``tiktok_count`` posts at $0.00207 per post.
- A Threads account is one request per run ($0.00207). An Instagram account or
  hashtag is up to 12 posts at the per-post price ($0.02484 per run) and a
  Facebook page up
  to 10 ($0.0207 per run). Meta ads are ``meta_ads_count`` Ad Library records
  per run: 10 is $0.0207, 30 is $0.0621.
- TikTok search has no working date filter and ranks by relevance, so a TikTok
  keyword monitor finds a new video only when it reaches TikTok's top results;
  to follow creators or a tag, watch their accounts (``tiktok_accounts``) or the
  hashtag (``tiktok_hashtags``). Threads search
  returns loosely matched posts, so ``threads_accounts`` work better there too.
- TikTok and Instagram can watch up to 10 hashtags each (``tiktok_hashtags``,
  ``instagram_hashtags``; any script, ``aversi``, ``#ავერსი`` or a tag URL). A
  hashtag feed lists posts from any account, so a monitor finds posts outside
  the accounts it watches. Items read from a hashtag have ``data.hashtag``, and a
  run lists hashtags it could not read in
  ``stats.sources.<source>.hashtags_failed``.
- Instagram and Facebook have no keyword search for posts: those sources read
  only watched feeds (``instagram_accounts`` or ``instagram_hashtags``,
  ``facebook_pages``), and the API answers 422 for either without any.
  Facebook has no organic hashtag feed. Meta ads match keywords, so a monitor with
  ``meta_ads`` needs a keyword query (a brand, product or short phrase); a
  sentence finds nothing.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from langchain_desearch.people import (
    PEOPLE_API_BASE_URL,
    PeopleClient,
    _compact,
    _drop_none,
    _list,
    _obj,
    _text,
)

MONITOR_API_BASE_URL = PEOPLE_API_BASE_URL

MONITOR_CREATE_TOOL = "monitor_create_tool"
MONITOR_ESTIMATE_TOOL = "monitor_estimate_tool"
MONITORS_TOOL = "monitors_tool"
MONITOR_UPDATE_TOOL = "monitor_update_tool"
MONITOR_CONTROL_TOOL = "monitor_control_tool"
MONITOR_RUNS_TOOL = "monitor_runs_tool"
MONITOR_ITEMS_TOOL = "monitor_items_tool"

MONITOR_SOURCES = (
    "web",
    "x",
    "reddit",
    "youtube",
    "threads",
    "tiktok",
    "instagram",
    "facebook",
    "meta_ads",
)
MONITOR_INTERVALS = ("1h", "3h", "6h", "12h", "1d", "7d")
MONITOR_EVENTS = ("monitor.items.found", "monitor.run.completed", "monitor.run.failed")
MONITOR_STATUSES = ("active", "paused")
MONITOR_CONTROL_ACTIONS = (
    "pause",
    "resume",
    "trigger",
    "delete",
    "test_webhook",
    "rotate_webhook_secret",
)

# Statuses after which a run never changes again (queued and running are the others).
MONITOR_RUN_ENDED_STATUSES = frozenset({"completed", "failed", "cancelled"})

MONITOR_WEB_COUNTS = (10, 15, 20)
MONITOR_TIKTOK_COUNTS = (10, 20, 30)
MONITOR_META_ADS_COUNTS = (10, 20, 30)
# Watched accounts and hashtags per source. The API cleans the names (``@OpenAI`` or
# an x.com URL becomes ``OpenAI``, ``r/LocalLLaMA`` becomes ``LocalLLaMA``, a Facebook
# ``profile.php?id=`` URL becomes the page ID, ``#ავერსი`` or a tag URL becomes the
# tag) and answers 422 for one it cannot read, so they are sent as given. It also
# answers 422 for instagram in ``sources`` without accounts or hashtags, or facebook
# without pages; on update it checks the monitor as patched, which only it knows, so
# that is left to the API too.
MONITOR_ACCOUNTS_MAX = 10
MONITOR_QUERY_MAX = 500
MONITOR_NAME_MAX = 120
MONITOR_CRITERIA_MAX = 5
MONITOR_CRITERION_MAX = 300
MONITOR_WEBHOOK_URL_MAX = 2000

# ``metadata`` labels: the API rejects anything else with HTTP 422.
MONITOR_METADATA_MAX_KEYS = 20
MONITOR_METADATA_KEY_MAX = 64
MONITOR_METADATA_VALUE_MAX = 500

_MAX_ITEM_TEXT_CHARS = 500

_FAIL_REASONS = {
    "insufficient_credits": (
        "the balance could not hold the run, so the monitor is paused until it is resumed"
    ),
    "all_sources_failed": "every source failed for this window",
    "worker_lost": "the worker running it stopped",
    "internal_error": "an internal error",
}


class MonitorClient:
    """Minimal async client for the Monitors routes."""

    def __init__(self, api_key: str, base_url: str = MONITOR_API_BASE_URL) -> None:
        # Same transport, auth header and error text as People Search.
        self._http = PeopleClient(api_key, base_url)

    async def __aenter__(self) -> "MonitorClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.close()

    @staticmethod
    def _path(monitor_id: str, suffix: str = "") -> str:
        return f"/desearch/monitors/{quote(str(monitor_id), safe='')}{suffix}"

    async def _data(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        _, data = await self._http._call(method, path, **kwargs)
        return data

    async def create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("POST", "/desearch/monitors", body=body)

    async def estimate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("POST", "/desearch/monitors/estimate", body=body)

    async def list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("GET", "/desearch/monitors", params=params)

    async def get(self, monitor_id: str) -> Dict[str, Any]:
        return await self._data("GET", self._path(monitor_id))

    async def update(self, monitor_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("PATCH", self._path(monitor_id), body=body)

    async def delete(self, monitor_id: str) -> Dict[str, Any]:
        return await self._data("DELETE", self._path(monitor_id))

    async def trigger(self, monitor_id: str) -> Dict[str, Any]:
        """Queue a billed run now. HTTP 409 while a run is queued or running."""
        return await self._data("POST", self._path(monitor_id, "/trigger"))

    async def list_runs(self, monitor_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("GET", self._path(monitor_id, "/runs"), params=params)

    async def get_run(self, monitor_id: str, run_id: str) -> Dict[str, Any]:
        return await self._data(
            "GET", self._path(monitor_id, f"/runs/{quote(str(run_id), safe='')}")
        )

    async def list_items(self, monitor_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._data("GET", self._path(monitor_id, "/items"), params=params)

    async def test_webhook(self, monitor_id: str) -> Dict[str, Any]:
        return await self._data("POST", self._path(monitor_id, "/webhook/test"))

    async def rotate_webhook_secret(self, monitor_id: str) -> Dict[str, Any]:
        return await self._data("POST", self._path(monitor_id, "/webhook/rotate-secret"))


def _check_choices(name: str, values: Optional[List[Any]], allowed: Tuple[Any, ...]) -> None:
    unknown = [value for value in values or [] if value not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {name}: {', '.join(map(str, unknown))}. "
            f"Use: {', '.join(map(str, allowed))}."
        )


def check_monitor_metadata(metadata: Optional[Dict[str, str]]) -> None:
    """The API's metadata rule for monitors, checked before the call."""
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object of string labels.")
    if len(metadata) > MONITOR_METADATA_MAX_KEYS:
        raise ValueError(f"metadata can have at most {MONITOR_METADATA_MAX_KEYS} keys.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not 1 <= len(key) <= MONITOR_METADATA_KEY_MAX:
            raise ValueError(
                f"metadata keys are 1-{MONITOR_METADATA_KEY_MAX} characters; "
                f"{str(key)[:50]!r} is not."
            )
        if not isinstance(value, str) or len(value) > MONITOR_METADATA_VALUE_MAX:
            raise ValueError(
                f"metadata value of {key!r} must be a string of at most "
                f"{MONITOR_METADATA_VALUE_MAX} characters."
            )


def monitor_settings(
    query: Optional[str] = None,
    name: Optional[str] = None,
    criteria: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    interval: Optional[str] = None,
    min_score: Optional[int] = None,
    summary: Optional[bool] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """The settings create and update share. Raises ValueError for a value the API would reject."""
    if query is not None:
        query = " ".join(query.split()) if isinstance(query, str) else query
        if not isinstance(query, str) or not 3 <= len(query) <= MONITOR_QUERY_MAX:
            raise ValueError(f"query must be 3-{MONITOR_QUERY_MAX} characters.")
    if name is not None and len(name) > MONITOR_NAME_MAX:
        raise ValueError(f"name must be at most {MONITOR_NAME_MAX} characters.")
    if criteria is not None:
        criteria = [item.strip() for item in criteria if isinstance(item, str) and item.strip()]
        if len(criteria) > MONITOR_CRITERIA_MAX:
            raise ValueError(f"criteria can list at most {MONITOR_CRITERIA_MAX} conditions.")
        if any(len(item) > MONITOR_CRITERION_MAX for item in criteria):
            raise ValueError(
                f"Each criterion must be at most {MONITOR_CRITERION_MAX} characters."
            )
    if sources is not None:
        sources = list(dict.fromkeys(sources))
        if not sources:
            raise ValueError(f"Pick at least one source: {', '.join(MONITOR_SOURCES)}.")
        _check_choices("sources", sources, MONITOR_SOURCES)
    if interval is not None and interval not in MONITOR_INTERVALS:
        raise ValueError(f"interval must be one of: {', '.join(MONITOR_INTERVALS)}.")
    if min_score is not None and not 0 <= min_score <= 100:
        raise ValueError("min_score must be between 0 and 100.")
    check_monitor_metadata(metadata)
    return _drop_none(
        {
            "query": query,
            "name": name,
            "criteria": criteria,
            "sources": sources,
            "interval": interval,
            "min_score": min_score,
            "summary": summary,
            "metadata": metadata,
        }
    )


def _check_accounts(name: str, accounts: Optional[List[str]], noun: str = "accounts") -> None:
    if accounts is None:
        return
    if len(accounts) > MONITOR_ACCOUNTS_MAX:
        raise ValueError(f"{name} can list at most {MONITOR_ACCOUNTS_MAX} {noun}.")
    if any(not isinstance(account, str) or not account.strip() for account in accounts):
        raise ValueError(f"{name} must list non-empty names.")


def monitor_options(
    web_count: Optional[int] = None,
    x_count: Optional[int] = None,
    x_lang: Optional[str] = None,
    x_min_likes: Optional[int] = None,
    x_accounts: Optional[List[str]] = None,
    reddit_subreddits: Optional[List[str]] = None,
    youtube_channels: Optional[List[str]] = None,
    tiktok_count: Optional[int] = None,
    tiktok_accounts: Optional[List[str]] = None,
    tiktok_hashtags: Optional[List[str]] = None,
    threads_accounts: Optional[List[str]] = None,
    instagram_accounts: Optional[List[str]] = None,
    instagram_hashtags: Optional[List[str]] = None,
    facebook_pages: Optional[List[str]] = None,
    meta_ads_count: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """The per-source ``options`` object, with only the sources that have a value set.

    An empty account or hashtag list is kept: on update it stops watching them (for
    instagram and facebook the API then answers 422 while the source has nothing
    left to watch).
    """
    if web_count is not None and web_count not in MONITOR_WEB_COUNTS:
        raise ValueError(
            f"web_count must be one of: {', '.join(map(str, MONITOR_WEB_COUNTS))}."
        )
    if x_count is not None and not 1 <= x_count <= 100:
        raise ValueError("x_count must be between 1 and 100.")
    if x_lang is not None and not 2 <= len(x_lang) <= 3:
        raise ValueError("x_lang must be a 2-3 letter language code, for example 'en'.")
    if x_min_likes is not None and x_min_likes < 0:
        raise ValueError("x_min_likes must be 0 or more.")
    if tiktok_count is not None and tiktok_count not in MONITOR_TIKTOK_COUNTS:
        raise ValueError(
            f"tiktok_count must be one of: {', '.join(map(str, MONITOR_TIKTOK_COUNTS))}."
        )
    if meta_ads_count is not None and meta_ads_count not in MONITOR_META_ADS_COUNTS:
        raise ValueError(
            f"meta_ads_count must be one of: {', '.join(map(str, MONITOR_META_ADS_COUNTS))}."
        )
    _check_accounts("x_accounts", x_accounts)
    _check_accounts("reddit_subreddits", reddit_subreddits)
    _check_accounts("youtube_channels", youtube_channels)
    _check_accounts("tiktok_accounts", tiktok_accounts)
    _check_accounts("tiktok_hashtags", tiktok_hashtags, "hashtags")
    _check_accounts("threads_accounts", threads_accounts)
    _check_accounts("instagram_accounts", instagram_accounts)
    _check_accounts("instagram_hashtags", instagram_hashtags, "hashtags")
    _check_accounts("facebook_pages", facebook_pages)
    options = {
        "web": _drop_none({"count": web_count}),
        "x": _drop_none(
            {"count": x_count, "lang": x_lang, "min_likes": x_min_likes, "accounts": x_accounts}
        ),
        "reddit": _drop_none({"subreddits": reddit_subreddits}),
        "youtube": _drop_none({"channels": youtube_channels}),
        "tiktok": _drop_none(
            {"count": tiktok_count, "accounts": tiktok_accounts, "hashtags": tiktok_hashtags}
        ),
        "threads": _drop_none({"accounts": threads_accounts}),
        "instagram": _drop_none(
            {"accounts": instagram_accounts, "hashtags": instagram_hashtags}
        ),
        "facebook": _drop_none({"pages": facebook_pages}),
        "meta_ads": _drop_none({"count": meta_ads_count}),
    }
    return {source: values for source, values in options.items() if values}


def monitor_webhook(url: str, events: Optional[List[str]] = None) -> Dict[str, Any]:
    """The ``webhook`` object. The API also refuses URLs on private addresses."""
    url = url.strip() if isinstance(url, str) else ""
    if not url.lower().startswith("https://"):
        raise ValueError("webhook_url must be an https:// URL on a public address.")
    if len(url) > MONITOR_WEBHOOK_URL_MAX:
        raise ValueError(f"webhook_url must be at most {MONITOR_WEBHOOK_URL_MAX} characters.")
    webhook: Dict[str, Any] = {"url": url}
    if events is not None:
        events = list(dict.fromkeys(events))
        if not events:
            raise ValueError(f"Pick at least one webhook event: {', '.join(MONITOR_EVENTS)}.")
        _check_choices("webhook_events", events, MONITOR_EVENTS)
        webhook["events"] = events
    return webhook


def monitor_create_body(
    query: str,
    name: Optional[str] = None,
    criteria: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    interval: Optional[str] = None,
    min_score: Optional[int] = None,
    summary: Optional[bool] = None,
    web_count: Optional[int] = None,
    x_count: Optional[int] = None,
    x_lang: Optional[str] = None,
    x_min_likes: Optional[int] = None,
    x_accounts: Optional[List[str]] = None,
    reddit_subreddits: Optional[List[str]] = None,
    youtube_channels: Optional[List[str]] = None,
    tiktok_count: Optional[int] = None,
    tiktok_accounts: Optional[List[str]] = None,
    tiktok_hashtags: Optional[List[str]] = None,
    threads_accounts: Optional[List[str]] = None,
    instagram_accounts: Optional[List[str]] = None,
    instagram_hashtags: Optional[List[str]] = None,
    facebook_pages: Optional[List[str]] = None,
    meta_ads_count: Optional[int] = None,
    webhook_url: Optional[str] = None,
    webhook_events: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    run_now: Optional[bool] = None,
) -> Dict[str, Any]:
    """Body of ``POST /desearch/monitors``. Raises ValueError for a value the API would reject."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required: what to watch, in plain words.")
    if webhook_events is not None and not webhook_url:
        raise ValueError("webhook_events needs a webhook_url.")
    body = monitor_settings(
        query, name, criteria, sources, interval, min_score, summary, metadata
    )
    options = monitor_options(
        web_count,
        x_count,
        x_lang,
        x_min_likes,
        x_accounts,
        reddit_subreddits,
        youtube_channels,
        tiktok_count,
        tiktok_accounts,
        tiktok_hashtags,
        threads_accounts,
        instagram_accounts,
        instagram_hashtags,
        facebook_pages,
        meta_ads_count,
    )
    if options:
        body["options"] = options
    if webhook_url:
        body["webhook"] = monitor_webhook(webhook_url, webhook_events)
    if run_now is not None:
        body["run_now"] = run_now
    return body


def monitor_estimate_body(
    sources: Optional[List[str]] = None,
    interval: Optional[str] = None,
    summary: Optional[bool] = None,
    web_count: Optional[int] = None,
    x_count: Optional[int] = None,
    x_lang: Optional[str] = None,
    x_min_likes: Optional[int] = None,
    x_accounts: Optional[List[str]] = None,
    reddit_subreddits: Optional[List[str]] = None,
    youtube_channels: Optional[List[str]] = None,
    tiktok_count: Optional[int] = None,
    tiktok_accounts: Optional[List[str]] = None,
    tiktok_hashtags: Optional[List[str]] = None,
    threads_accounts: Optional[List[str]] = None,
    instagram_accounts: Optional[List[str]] = None,
    instagram_hashtags: Optional[List[str]] = None,
    facebook_pages: Optional[List[str]] = None,
    meta_ads_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Body of ``POST /desearch/monitors/estimate``. Watched accounts and hashtags change the price.

    The estimate does not require accounts or hashtags for instagram or facebook (it prices one).
    """
    body = monitor_settings(sources=sources, interval=interval, summary=summary)
    options = monitor_options(
        web_count,
        x_count,
        x_lang,
        x_min_likes,
        x_accounts,
        reddit_subreddits,
        youtube_channels,
        tiktok_count,
        tiktok_accounts,
        tiktok_hashtags,
        threads_accounts,
        instagram_accounts,
        instagram_hashtags,
        facebook_pages,
        meta_ads_count,
    )
    if options:
        body["options"] = options
    return body


def monitor_list_params(
    status: Optional[List[str]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """Query parameters for ``GET /desearch/monitors``."""
    if isinstance(status, str):
        status = status.split(",")
    statuses = list(dict.fromkeys(s.strip() for s in status or [] if s and s.strip()))
    _check_choices("status", statuses, MONITOR_STATUSES)
    return _drop_none(
        {"status": ",".join(statuses) or None, "limit": limit, "cursor": cursor}
    )


def _truncated(value: Any, limit: int) -> Optional[str]:
    text = _text(value)
    if text and len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _compact_webhook(webhook: Dict[str, Any]) -> Dict[str, Any]:
    """The webhook with its newest delivery (``last_delivery``) as the API returns it.

    ``last_delivery`` is null before the first delivery. It is kept as null, so a
    reader can tell "nothing delivered yet" from an API that does not report it.
    """
    view = _compact(
        {
            "url": webhook.get("url"),
            "events": _list(webhook.get("events")),
            "consecutive_failures": webhook.get("consecutive_failures"),
        }
    )
    if view and "last_delivery" in webhook:
        delivery = webhook["last_delivery"]
        view["last_delivery"] = delivery if isinstance(delivery, dict) else None
    return view


def compact_monitor(monitor: Any) -> Dict[str, Any]:
    """A monitor as an agent needs it: what it watches, when it runs, its state and cost."""
    m = _obj(monitor)
    webhook = _obj(m.get("webhook"))
    last_run = _obj(m.get("last_run"))
    return _compact(
        {
            "monitor_id": m.get("monitor_id"),
            "name": _text(m.get("name")),
            "query": m.get("query"),
            "criteria": _list(m.get("criteria")),
            "sources": _list(m.get("sources")),
            "interval": m.get("interval"),
            "options": _obj(m.get("options")),
            "min_score": m.get("min_score"),
            "summary": m.get("summary"),
            "status": m.get("status"),
            "paused_reason": m.get("paused_reason"),
            "webhook": _compact_webhook(webhook),
            "webhook_secret": _text(m.get("webhook_secret")),
            "metadata": _obj(m.get("metadata")),
            "max_run_cost_usd": m.get("max_run_cost_usd"),
            "next_run_at": m.get("next_run_at"),
            "last_run": _compact(
                {
                    "run_id": last_run.get("run_id"),
                    "status": last_run.get("status"),
                    "item_count": last_run.get("item_count"),
                    "cost_usd": last_run.get("cost_usd"),
                    "fail_reason": last_run.get("fail_reason"),
                    "finished_at": last_run.get("finished_at"),
                }
            ),
            "stats": _obj(m.get("stats")),
            "created_at": m.get("created_at"),
            "updated_at": m.get("updated_at"),
        }
    )


def compact_run(run: Any) -> Dict[str, Any]:
    """Run fields an agent needs: status, window, what each source returned, items kept and cost."""
    r = _obj(run)
    status = r.get("status")
    return _compact(
        {
            "run_id": r.get("run_id"),
            "monitor_id": r.get("monitor_id"),
            "trigger": r.get("trigger"),
            "status": status,
            "done": status in MONITOR_RUN_ENDED_STATUSES if isinstance(status, str) else None,
            "fail_reason": r.get("fail_reason"),
            "window_start": r.get("window_start"),
            "window_end": r.get("window_end"),
            "item_count": r.get("item_count"),
            "stats": _obj(r.get("stats")),
            "summary": _text(r.get("summary")),
            "reserved_usd": r.get("reserved_usd"),
            "cost_usd": r.get("cost_usd"),
            "created_at": r.get("created_at"),
            "finished_at": r.get("finished_at"),
        }
    )


def compact_item(item: Any) -> Dict[str, Any]:
    """One kept item: where it is, what it says, who posted it and how well it fits."""
    i = _obj(item)
    author = _obj(i.get("author"))
    return _compact(
        {
            "item_id": i.get("item_id"),
            "run_id": i.get("run_id"),
            "source": i.get("source"),
            "url": _text(i.get("url")),
            "title": _text(i.get("title")),
            "text": _truncated(i.get("text"), _MAX_ITEM_TEXT_CHARS),
            "author": _compact(
                {
                    "name": _text(author.get("name")),
                    "handle": _text(author.get("handle")),
                    "url": _text(author.get("url")),
                }
            ),
            "published_at": i.get("published_at"),
            "score": i.get("score"),
            "signals": _obj(i.get("signals")),
            "data": _compact(_obj(i.get("data"))),
            "found_at": i.get("found_at"),
        }
    )


def _money(value: Any, fallback: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return f"${value:.4f}"


def monitor_next_step(monitor: Dict[str, Any], first_run_queued: bool = False) -> str:
    """What the monitor will do next and where to read what it finds."""
    monitor_id = monitor.get("monitor_id")
    read = (
        f"Read what it keeps with {MONITOR_ITEMS_TOOL} (monitor_id='{monitor_id}') and its "
        f"runs with {MONITOR_RUNS_TOOL}; both are free."
    )
    status = monitor.get("status")
    reason = monitor.get("paused_reason")
    resume = f"call {MONITOR_CONTROL_TOOL} with monitor_id='{monitor_id}' and action='resume'"
    if status == "paused" and reason == "insufficient_credits":
        step = f"Paused because the balance could not hold a run. Add credits, then {resume}."
    elif status == "paused" and reason == "webhook_failing":
        step = (
            "Paused because 5 webhook deliveries in a row failed. Fix the endpoint (check it "
            f"with action='test_webhook'), then {resume}."
        )
    elif status == "paused":
        step = (
            f"Paused: it does not run on schedule until you {resume}. action='trigger' still "
            "runs it once (billed)."
        )
    else:
        when = f"; next run at {monitor['next_run_at']}" if monitor.get("next_run_at") else ""
        step = (
            f"Active: it runs every {monitor.get('interval') or '1d'}{when}. Each run is billed, "
            f"up to {_money(monitor.get('max_run_cost_usd'), 'its max_run_cost_usd')}."
        )
        if first_run_queued:
            step = f"The first run is queued now. {step}"
    step = f"{step} {read}"
    if monitor.get("webhook_secret"):
        step += (
            " Store webhook_secret now: it is returned only once. Verify each delivery's "
            "Desearch-Signature header with it."
        )
    return step


def run_next_step(run: Dict[str, Any]) -> str:
    """What a run did, or how to keep waiting on it."""
    monitor_id = run.get("monitor_id")
    run_id = run.get("run_id")
    status = run.get("status")
    if status == "completed":
        count = run.get("item_count") or 0
        kept = f"kept {count} new item{'' if count == 1 else 's'}" if count else "kept no new item"
        cost = _money(run.get("cost_usd"), "")
        billed = f" and was billed {cost}" if cost else ""
        return f"Completed: it {kept}{billed}."
    if status == "failed":
        reason = run.get("fail_reason")
        why = _FAIL_REASONS.get(reason, reason or "an unknown reason")
        return f"Failed: {why}."
    if status == "cancelled":
        return "Cancelled."
    return (
        f"Still {status or 'queued'}. Call {MONITOR_RUNS_TOOL} again with monitor_id="
        f"'{monitor_id}', run_id='{run_id}' and a timeout to wait for it; that is free."
    )


async def create_monitor(client: MonitorClient, *, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create a monitor and say what happens next."""
    monitor = _obj(await client.create(body))
    output = compact_monitor(monitor)
    output["next_step"] = monitor_next_step(monitor, first_run_queued=body.get("run_now", True))
    return output


async def estimate_monitor(client: MonitorClient, *, body: Dict[str, Any]) -> Dict[str, Any]:
    """The most one run and one month of runs can cost."""
    return _obj(await client.estimate(body))


async def get_monitors(
    client: MonitorClient,
    *,
    monitor_id: Optional[str] = None,
    status: Optional[List[str]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """One monitor by ``monitor_id``, or one page of the caller's monitors, newest first."""
    if monitor_id:
        monitor = _obj(await client.get(monitor_id))
        output = compact_monitor(monitor)
        output["next_step"] = monitor_next_step(monitor)
        return output
    page = _obj(await client.list(monitor_list_params(status, limit, cursor)))
    output: Dict[str, Any] = {
        "monitors": [compact_monitor(monitor) for monitor in _list(page.get("monitors"))]
    }
    next_cursor = page.get("next_cursor")
    if next_cursor:
        output["next_cursor"] = next_cursor
        output["next_step"] = (
            f"More monitors match. Call {MONITORS_TOOL} again with the same status and "
            f"cursor='{next_cursor}'."
        )
    return output


def _merged_options(
    current: Dict[str, Any], changes: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """The monitor's options with ``changes`` applied: PATCH replaces the whole object.

    Each changed field replaces the current one, so a new account or hashtag list
    replaces the watched accounts or hashtags of that source; every other field and
    source is kept.
    """
    merged: Dict[str, Any] = {}
    for source, values in _obj(current).items():
        if isinstance(values, dict):
            merged[source] = _drop_none(values)
    for source, values in changes.items():
        merged[source] = {**merged.get(source, {}), **values}
    return merged


async def update_monitor(
    client: MonitorClient,
    *,
    monitor_id: str,
    settings: Dict[str, Any],
    options: Dict[str, Dict[str, Any]],
    webhook_url: Optional[str] = None,
    webhook_events: Optional[List[str]] = None,
    remove_webhook: bool = False,
) -> Dict[str, Any]:
    """Change a monitor's settings. Options and webhook events are merged with the current ones."""
    body = dict(settings)
    if remove_webhook:
        body["webhook"] = None
    elif options or webhook_url or webhook_events:
        # PATCH replaces options and the webhook as a whole, so start from what is set now.
        current = _obj(await client.get(monitor_id))
        if options:
            body["options"] = _merged_options(_obj(current.get("options")), options)
        if webhook_url or webhook_events:
            hook = _obj(current.get("webhook"))
            url = webhook_url or _text(hook.get("url"))
            if not url:
                raise ValueError("This monitor has no webhook. Set webhook_url to add one.")
            events = webhook_events or _list(hook.get("events")) or None
            body["webhook"] = monitor_webhook(url, events)
    if not body:
        raise ValueError("Nothing to change: set at least one setting.")
    monitor = _obj(await client.update(monitor_id, body))
    output = compact_monitor(monitor)
    output["next_step"] = monitor_next_step(monitor)
    return output


async def control_monitor(
    client: MonitorClient, *, monitor_id: str, action: str
) -> Dict[str, Any]:
    """Pause, resume, trigger or delete a monitor, or test or rotate its webhook."""
    if action not in MONITOR_CONTROL_ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(MONITOR_CONTROL_ACTIONS)}.")
    if action in ("pause", "resume"):
        status = "paused" if action == "pause" else "active"
        monitor = _obj(await client.update(monitor_id, {"status": status}))
        output = compact_monitor(monitor)
        output["next_step"] = monitor_next_step(monitor)
        return output
    if action == "trigger":
        run = _obj(await client.trigger(monitor_id))
        output = compact_run(run)
        hold = _money(run.get("reserved_usd"), "the monitor's max_run_cost_usd")
        output["next_step"] = (
            f"Run queued; it is billed like a scheduled run, up to {hold}. Call "
            f"{MONITOR_RUNS_TOOL} with monitor_id='{monitor_id}', run_id="
            f"'{run.get('run_id')}' and timeout=120 to wait for the items it keeps."
        )
        return output
    if action == "delete":
        output = _obj(await client.delete(monitor_id))
        return {**output, "next_step": "Deleted: it will not run again."}
    if action == "test_webhook":
        result = _obj(await client.test_webhook(monitor_id))
        if result.get("delivered"):
            step = f"Delivered: your endpoint answered HTTP {result.get('status_code')}."
        else:
            answer = result.get("error") or f"HTTP {result.get('status_code')}"
            step = (
                f"Not delivered ({answer}). Real deliveries need a 2xx within 10 seconds and "
                "are retried after 1, 5, 30, 120 and 360 minutes."
            )
        return {**result, "next_step": step}
    result = _obj(await client.rotate_webhook_secret(monitor_id))
    return {
        **result,
        "next_step": (
            "Store webhook_secret now: it is returned only once, and deliveries from now on "
            "are signed with it."
        ),
    }


async def read_runs(
    client: MonitorClient,
    *,
    monitor_id: str,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    timeout: float = 0,
    poll_interval: float = 5.0,
) -> Dict[str, Any]:
    """One run with the items it kept (waiting up to ``timeout`` seconds), or a page of runs."""
    if not run_id:
        page = _obj(
            await client.list_runs(monitor_id, _drop_none({"limit": limit, "cursor": cursor}))
        )
        output: Dict[str, Any] = {"runs": [compact_run(run) for run in _list(page.get("runs"))]}
        next_cursor = page.get("next_cursor")
        if next_cursor:
            output["next_cursor"] = next_cursor
            output["next_step"] = (
                f"More runs. Call {MONITOR_RUNS_TOOL} again with monitor_id='{monitor_id}' "
                f"and cursor='{next_cursor}'."
            )
        return output

    run = _obj(await client.get_run(monitor_id, run_id))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(float(timeout), 0.0)
    while run.get("status") not in MONITOR_RUN_ENDED_STATUSES:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(poll_interval, remaining))
        run = _obj(await client.get_run(monitor_id, run_id))
    output = compact_run(run)
    output.setdefault("monitor_id", monitor_id)
    output.setdefault("run_id", run_id)
    output["items"] = [compact_item(item) for item in _list(run.get("items"))]
    output["next_step"] = run_next_step(output)
    return output


async def read_items(
    client: MonitorClient,
    *,
    monitor_id: str,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """One page of everything the monitor kept, newest first."""
    if source is not None and source not in MONITOR_SOURCES:
        raise ValueError(f"source must be one of: {', '.join(MONITOR_SOURCES)}.")
    params = _drop_none({"source": source, "limit": limit, "cursor": cursor})
    page = _obj(await client.list_items(monitor_id, params))
    output: Dict[str, Any] = {
        "items": [compact_item(item) for item in _list(page.get("items"))]
    }
    next_cursor = page.get("next_cursor")
    if next_cursor:
        output["next_cursor"] = next_cursor
        output["next_step"] = (
            f"Older items. Call {MONITOR_ITEMS_TOOL} again with monitor_id='{monitor_id}', the "
            f"same source and cursor='{next_cursor}'."
        )
    return output
