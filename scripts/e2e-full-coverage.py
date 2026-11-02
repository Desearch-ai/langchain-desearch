#!/usr/bin/env python3
"""Live smoke test of the tools added in langchain-desearch 1.2.0.

Calls every new tool once against the live Desearch API through ``invoke``
(and one call through ``ainvoke``) with the smallest counts, follows one
cursor page per social platform when a cursor comes back, and prints one line
per call: ``PASS <name>``, ``FAIL <name>: <error>`` or ``SKIP <name>: <why>``.
Exits 1 if any call fails.

Each call is billed at the normal route price. People Search is a billed
background job, so it runs only with ``--with-people``.

    DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py
    DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py --with-people

Needs desearch-py 1.8.0 installed. Not run by CI.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Run from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_desearch import (  # noqa: E402
    AIWebLinksSearchTool,
    AIXLinksSearchTool,
    FacebookTool,
    FetchTwitterTrendsTool,
    InstagramTool,
    PeopleProfilesTool,
    PeopleSearchTool,
    SearchApiTool,
    SearchTweetsByUserTool,
    TikTokTool,
    WebExtractTool,
)

ERROR_PREFIX = "An error occurred while calling Desearch"

# Public accounts and tags that exist.
TIKTOK_USER = "tiktok"
INSTAGRAM_USER = "instagram"
FACEBOOK_PAGE = "facebook"
X_USER = "elonmusk"
HASHTAG = "ai"


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
            problem = _error(result) or (check(result) if check else None)
        except Exception as exc:  # report and keep going
            result, problem = None, f"{type(exc).__name__}: {exc}"
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


def _one_line(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _error(result: Any) -> Optional[str]:
    if result is None:
        return "no result"
    if isinstance(result, str) and result.startswith(ERROR_PREFIX):
        return result
    return None


def _is_dict(result: Any) -> Optional[str]:
    return (
        None
        if isinstance(result, dict)
        else f"expected an object, got {type(result).__name__}"
    )


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


def _first(page: Any, key: str) -> Optional[str]:
    """``key`` of the first result on a page that has it."""
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
) -> Any:
    return smoke.run(
        f"{tool.name} {action}", lambda: tool.invoke({"action": action, **args}), check
    )


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


def check_web_and_x(smoke: Smoke) -> None:
    smoke.run(
        "search_api_tool",
        lambda: SearchApiTool().invoke(
            {"query": "Bittensor decentralized AI", "mode": "fast", "count": 10}
        ),
        _has_key("results"),
    )
    smoke.run(
        "web_extract_tool",
        lambda: WebExtractTool().invoke({"url": "https://desearch.ai"}),
        lambda text: None if isinstance(text, str) and text.strip() else "empty page",
    )
    smoke.run(
        "ai_web_links_search_tool",
        lambda: AIWebLinksSearchTool().invoke(
            {"prompt": "Bittensor subnet 22 Desearch", "count": 10}
        ),
        _has_key("search_results"),
    )
    smoke.run(
        "ai_x_links_search_tool",
        lambda: AIXLinksSearchTool().invoke({"prompt": "Bittensor", "count": 10}),
        _has_key("miner_tweets"),
    )
    smoke.run(
        "search_tweets_by_user_tool",
        lambda: SearchTweetsByUserTool().invoke({"user": X_USER, "count": 1}),
        lambda r: None if isinstance(r, (list, dict)) else f"got {type(r).__name__}",
    )
    smoke.run(
        "fetch_twitter_trends_tool",
        lambda: FetchTwitterTrendsTool().invoke({"woeid": 1, "count": 30}),
        _has_key("trends"),
    )


def check_tiktok(smoke: Smoke) -> None:
    tool = TikTokTool()
    search = _social(smoke, tool, "search", {"query": HASHTAG, "count": 3}, _is_page)
    hashtag_args = {"tag": HASHTAG, "count": 3}
    hashtag = _social(smoke, tool, "hashtag", hashtag_args, _is_page)
    _follow_cursor(smoke, tool, "hashtag", hashtag_args, hashtag)
    _social(smoke, tool, "profile", {"username": TIKTOK_USER}, _is_dict)
    posts = _social(
        smoke, tool, "profile_posts", {"username": TIKTOK_USER, "count": 2}, _is_page
    )
    _social(smoke, tool, "trending", {"region": "US", "count": 3}, _is_page)

    video_id = _first(search, "id") or _first(hashtag, "id") or _first(posts, "id")
    if video_id is None:
        smoke.fail(
            "tiktok_tool post",
            "no video id: search, hashtag and profile_posts returned no posts",
        )
        smoke.fail("tiktok_tool comments", "no video id to read comments for")
        return
    _social(smoke, tool, "post", {"id": video_id}, _is_dict)
    _social(smoke, tool, "comments", {"video_id": video_id, "count": 3}, _is_page)


def check_facebook(smoke: Smoke) -> None:
    tool = FacebookTool()
    _social(smoke, tool, "search", {"query": HASHTAG, "count": 2}, _is_page)
    _social(smoke, tool, "hashtag", {"tag": HASHTAG, "count": 2}, _is_page)
    _social(smoke, tool, "profile", {"profile_id": FACEBOOK_PAGE}, _is_dict)
    posts_args = {"profile_id": FACEBOOK_PAGE, "count": 3}
    posts = _social(smoke, tool, "profile_posts", posts_args, _is_page)
    _follow_cursor(smoke, tool, "profile_posts", posts_args, posts)

    post_id = _first(posts, "id")
    if post_id is None:
        smoke.fail("facebook_tool post", "no post id: profile_posts returned no posts")
        smoke.fail("facebook_tool comments", "no post id to read comments for")
        return
    _social(smoke, tool, "post", {"id": post_id}, _is_dict)
    _social(smoke, tool, "comments", {"post_id": post_id, "count": 3}, _is_page)


def check_instagram(smoke: Smoke) -> None:
    tool = InstagramTool()
    _social(smoke, tool, "search", {"query": INSTAGRAM_USER, "count": 3}, _is_page)
    _social(smoke, tool, "hashtag", {"tag": HASHTAG, "count": 3}, _is_page)
    _social(smoke, tool, "profile", {"username": INSTAGRAM_USER}, _is_dict)
    posts_args = {"username": INSTAGRAM_USER, "count": 3}
    posts = _social(smoke, tool, "profile_posts", posts_args, _is_page)
    _follow_cursor(smoke, tool, "profile_posts", posts_args, posts)

    shortcode = _first(posts, "shortcode")
    if shortcode is None:
        smoke.fail(
            "instagram_tool post", "no shortcode: profile_posts returned no posts"
        )
        smoke.fail("instagram_tool comments", "no shortcode to read comments for")
        return
    _social(smoke, tool, "post", {"id": shortcode}, _is_dict)
    _social(smoke, tool, "comments", {"shortcode": shortcode, "count": 3}, _is_page)


def check_async(smoke: Smoke) -> None:
    smoke.run(
        "tiktok_tool profile (ainvoke)",
        lambda: asyncio.run(
            TikTokTool().ainvoke({"action": "profile", "username": TIKTOK_USER})
        ),
        _is_dict,
    )


def check_people(smoke: Smoke) -> None:
    smoke.run(
        "people_profiles_tool",
        lambda: PeopleProfilesTool().invoke({"platform": "x", "limit": 3}),
        _has_key("people"),
    )
    # Billed: reserves 5 profiles at the route price and settles to those delivered.
    smoke.run(
        "people_search_tool",
        lambda: PeopleSearchTool().invoke(
            {
                "query": "Founders of AI search startups who post about search APIs",
                "platform": "x",
                "count": 5,
                "timeout": 60,
            }
        ),
        _has_key("task_id"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--with-people",
        action="store_true",
        help="also start one billed People Search (count 5) and list saved profiles",
    )
    args = parser.parse_args()

    if not os.getenv("DESEARCH_API_KEY"):
        print("DESEARCH_API_KEY is not set.", file=sys.stderr)
        return 2

    smoke = Smoke()
    check_web_and_x(smoke)
    check_tiktok(smoke)
    check_facebook(smoke)
    check_instagram(smoke)
    check_async(smoke)
    if args.with_people:
        check_people(smoke)
    else:
        smoke.skip("people_search_tool", "billed job; pass --with-people to run it")

    print(f"{smoke.passed} passed, {smoke.failed} failed, {smoke.skipped} skipped")
    return 1 if smoke.failed else 0


if __name__ == "__main__":
    sys.exit(main())
