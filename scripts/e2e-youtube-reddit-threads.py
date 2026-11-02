#!/usr/bin/env python3
"""Live smoke test of the YouTube, Reddit and Threads tools (langchain-desearch 1.3.0).

Calls every action of ``youtube_tool``, ``reddit_tool`` and ``threads_tool``
once through ``invoke`` (one call through ``ainvoke``), follows one cursor
page on each paged action, and prints one line per call: ``PASS <name>``,
``FAIL <name>: <error>`` or ``SKIP <name>: <why>``. Exits 1 if any call fails.

Each call is billed as one request ($0.00207); a run is about 22 requests
(about $0.05). ``--base-url`` points the tools at another deployment, such as
https://api-dev.desearch.ai, and ``--key-env`` names the variable that holds
its key. Keys are read from the environment or ``~/.config/desearch/e2e.env``.

    DESEARCH_API_KEY=... python scripts/e2e-youtube-reddit-threads.py
    python scripts/e2e-youtube-reddit-threads.py --base-url https://api-dev.desearch.ai --key-env DESEARCH_DEV_API_KEY

Needs desearch-py 1.11.0 installed. Not run by CI.
"""

import argparse
import asyncio
import functools
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Run from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import langchain_desearch.tools as tools_module  # noqa: E402
from langchain_desearch import RedditTool, ThreadsTool, YouTubeTool  # noqa: E402

ERROR_PREFIX = "An error occurred while calling Desearch"

# Public targets that exist.
QUERY = "bittensor"
YOUTUBE_VIDEO_ID = "dQw4w9WgXcQ"
YOUTUBE_CHANNEL = "@MrBeast"
SUBREDDIT = "AskReddit"
FALLBACK_REDDIT_POST_ID = "1q6pxwn"
THREADS_USER = "zuck"
FALLBACK_THREADS_CODE = "DK-BydcJHkF"


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
        """Run one call, retried once after a 5xx (a known transient upstream 503).

        ``check`` returns a problem, or None if the result is fine.
        """
        for attempt in (1, 2):
            try:
                result = call()
                problem = _error(result) or (check(result) if check else None)
            except Exception as exc:  # report and keep going
                result, problem = None, f"{type(exc).__name__}: {exc}"
            if problem and attempt == 1 and _is_server_error(problem):
                print(f"RETRY {name}: {_one_line(problem, 120)}", flush=True)
                continue
            break
        if problem:
            self.fail(name, problem)
            return None
        print(f"PASS {name}", flush=True)
        self.passed += 1
        return result

    def fail(self, name: str, problem: str) -> None:
        print(f"FAIL {name}: {_one_line(problem)}", flush=True)
        self.failed += 1

    def skip(self, name: str, why: str) -> None:
        print(f"SKIP {name}: {why}", flush=True)
        self.skipped += 1


def _is_server_error(problem: str) -> bool:
    # desearch-py errors read "<status>, message=...".
    return re.search(r"Desearch: 5\d\d, message=", problem) is not None


def _one_line(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _error(result: Any) -> Optional[str]:
    if result is None:
        return "no result"
    if isinstance(result, str) and result.startswith(ERROR_PREFIX):
        return result
    return None


def _has_key(key: str) -> Callable[[Any], Optional[str]]:
    def check(result: Any) -> Optional[str]:
        if not isinstance(result, dict) or key not in result:
            return f"expected {key!r} in the result, got {_one_line(repr(result), 200)}"
        return None

    return check


def _is_page(result: Any) -> Optional[str]:
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        return f"expected {{results, next_cursor}}, got {_one_line(repr(result), 200)}"
    if "next_cursor" not in result:
        return "next_cursor is missing"
    return None


def _is_last_page(result: Any) -> Optional[str]:
    problem = _is_page(result)
    if problem is None and result["next_cursor"] is not None:
        return "Threads should not return a cursor"
    return problem


def _first(page: Any, key: str) -> Optional[str]:
    if not isinstance(page, dict):
        return None
    for item in page.get("results") or []:
        if isinstance(item, dict) and item.get(key):
            return str(item[key])
    return None


def _social(
    smoke: Smoke,
    tool: Any,
    action: str,
    args: Dict[str, Any],
    check: Callable[[Any], Optional[str]],
    label: str = "",
) -> Any:
    name = f"{tool.name} {action}" + (f" ({label})" if label else "")
    return smoke.run(name, lambda: tool.invoke({"action": action, **args}), check)


def _follow_cursor(
    smoke: Smoke, tool: Any, action: str, args: Dict[str, Any], page: Any
) -> None:
    name = f"{tool.name} {action} (next page)"
    cursor = page.get("next_cursor") if isinstance(page, dict) else None
    if not cursor:
        smoke.skip(name, "no next_cursor came back on the first page")
        return
    smoke.run(
        name,
        lambda: tool.invoke({"action": action, **args, "cursor": cursor}),
        _is_page,
    )


def check_youtube(smoke: Smoke) -> None:
    tool = YouTubeTool()
    search_args = {"query": QUERY, "type": "videos"}
    search = _social(smoke, tool, "search", search_args, _is_page)
    _follow_cursor(smoke, tool, "search", search_args, search)
    _social(smoke, tool, "video", {"id": YOUTUBE_VIDEO_ID}, _has_key("title"))
    _social(
        smoke,
        tool,
        "video",
        {"id": f"https://youtu.be/{YOUTUBE_VIDEO_ID}"},
        _has_key("title"),
        "URL id",
    )
    _social(
        smoke,
        tool,
        "transcript",
        {"video_id": YOUTUBE_VIDEO_ID, "language": "en"},
        _has_key("segments"),
    )
    comments_args = {"video_id": YOUTUBE_VIDEO_ID, "order": "top"}
    comments = _social(smoke, tool, "comments", comments_args, _is_page)
    _follow_cursor(smoke, tool, "comments", comments_args, comments)
    _social(smoke, tool, "channel", {"handle": YOUTUBE_CHANNEL}, _has_key("subscribers"))
    videos_args = {"handle": YOUTUBE_CHANNEL, "sort": "latest"}
    videos = _social(smoke, tool, "channel_videos", videos_args, _is_page)
    _follow_cursor(smoke, tool, "channel_videos", videos_args, videos)


def check_reddit(smoke: Smoke) -> None:
    tool = RedditTool()
    search_args = {"query": QUERY}
    search = _social(smoke, tool, "search", search_args, _is_page)
    _follow_cursor(smoke, tool, "search", search_args, search)
    _social(smoke, tool, "subreddit", {"name": SUBREDDIT}, _has_key("name"))
    posts_args = {"name": f"r/{SUBREDDIT}", "sort": "top", "timeframe": "week"}
    posts = _social(smoke, tool, "subreddit_posts", posts_args, _is_page)
    _follow_cursor(smoke, tool, "subreddit_posts", posts_args, posts)
    post_id = _first(search, "id") or FALLBACK_REDDIT_POST_ID
    _social(smoke, tool, "post", {"id": post_id}, _has_key("title"))
    comments_args = {"post_id": post_id}
    comments = _social(smoke, tool, "comments", comments_args, _is_page)
    if isinstance(comments, dict) and comments.get("next_cursor"):
        _follow_cursor(smoke, tool, "comments", comments_args, comments)


def check_threads(smoke: Smoke) -> None:
    tool = ThreadsTool()
    search = _social(smoke, tool, "search", {"query": QUERY}, _is_last_page)
    _social(smoke, tool, "search_users", {"query": THREADS_USER}, _is_last_page)
    _social(smoke, tool, "profile", {"username": f"@{THREADS_USER}"}, _has_key("username"))
    posts = _social(
        smoke, tool, "profile_posts", {"username": THREADS_USER}, _is_last_page
    )
    code = _first(posts, "code") or _first(search, "code") or FALLBACK_THREADS_CODE
    _social(smoke, tool, "post", {"id": code}, _has_key("comments"))


def check_async(smoke: Smoke) -> None:
    smoke.run(
        "threads_tool profile (ainvoke)",
        lambda: asyncio.run(
            ThreadsTool().ainvoke({"action": "profile", "username": THREADS_USER})
        ),
        _has_key("username"),
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
    args = parser.parse_args()

    _load_env_file()
    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key:
        print(f"{args.key_env} is not set.", file=sys.stderr)
        return 2
    # The tools read DESEARCH_API_KEY and build the SDK client with its default
    # base URL; point both at the deployment under test.
    os.environ["DESEARCH_API_KEY"] = api_key
    tools_module.Desearch = functools.partial(tools_module.Desearch, base_url=args.base_url)

    smoke = Smoke()
    check_youtube(smoke)
    check_reddit(smoke)
    check_threads(smoke)
    check_async(smoke)

    print(
        f"{smoke.passed} passed, {smoke.failed} failed, {smoke.skipped} skipped "
        f"on {args.base_url}"
    )
    return 1 if smoke.failed else 0


if __name__ == "__main__":
    sys.exit(main())
