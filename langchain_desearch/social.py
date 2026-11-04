"""TikTok, Facebook, Instagram, YouTube, Reddit and Threads actions for the social tools.

Each platform has one LangChain tool with an ``action`` field instead of one
tool per endpoint, so an agent's tool list stays short. This module maps every
action to its desearch-py method, checks that the call has the fields the
action needs (and none it would ignore), and shapes list responses as
``{"results": [...], "next_cursor": ...}``.

List routes put the next-page cursor in the ``x-desearch-next-cursor``
response header. desearch-py returns it as ``metadata.next_cursor`` when it is
called with ``include_metadata=True``. The cursor is opaque: it is passed back
exactly as received, with the same action and target.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SOCIAL_PLATFORMS = (
    "tiktok",
    "facebook",
    "instagram",
    "youtube",
    "reddit",
    "threads",
)


@dataclass(frozen=True)
class SocialAction:
    """One action: the SDK method it calls and the fields it takes."""

    method: str
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    # True when the API answers with a JSON array (one page of items).
    returns_list: bool = False
    # Fields that only take one of a few values: {field: allowed values}.
    choices: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    @property
    def fields(self) -> Tuple[str, ...]:
        return self.required + self.optional

    @property
    def paginates(self) -> bool:
        return "cursor" in self.optional


_PAGE = ("count", "cursor")
_TIMEFRAMES = ("all", "day", "week", "month", "year")

SOCIAL_ACTIONS: Dict[str, Dict[str, SocialAction]] = {
    "tiktok": {
        "search": SocialAction(
            "tiktok_search", ("query",), ("count", "cursor", "sort"), True
        ),
        "post": SocialAction("tiktok_post_by_id", ("id",)),
        "profile": SocialAction("tiktok_profile", ("username",)),
        "profile_posts": SocialAction(
            "tiktok_profile_posts", ("username",), _PAGE, True
        ),
        "comments": SocialAction("tiktok_comments", ("video_id",), _PAGE, True),
        "hashtag": SocialAction("tiktok_hashtag", ("tag",), _PAGE, True),
        "trending": SocialAction("tiktok_trending", (), ("region", "count"), True),
    },
    "facebook": {
        # sort: impressions (the API default, most seen first) or recent (recently
        # started ads first).
        "search": SocialAction(
            "facebook_search",
            ("query",),
            ("count", "cursor", "sort"),
            True,
            (("sort", ("impressions", "recent")),),
        ),
        "post": SocialAction("facebook_post_by_id", ("id",)),
        "profile": SocialAction("facebook_profile", ("profile_id",)),
        "profile_posts": SocialAction(
            "facebook_profile_posts", ("profile_id",), _PAGE, True
        ),
        "comments": SocialAction("facebook_comments", ("post_id",), _PAGE, True),
        "hashtag": SocialAction("facebook_hashtag", ("tag",), _PAGE, True),
    },
    "instagram": {
        "search": SocialAction("instagram_search", ("query",), ("count",), True),
        "post": SocialAction("instagram_media", ("id",)),
        "profile": SocialAction("instagram_profile", ("username",)),
        "profile_posts": SocialAction(
            "instagram_profile_posts", ("username",), _PAGE, True
        ),
        "comments": SocialAction("instagram_comments", ("shortcode",), _PAGE, True),
        "hashtag": SocialAction("instagram_hashtag", ("tag",), _PAGE, True),
    },
    # YouTube, Reddit and Threads routes take no count: a page is whatever the
    # platform returns. Threads never pages.
    "youtube": {
        "search": SocialAction(
            "youtube_search",
            ("query",),
            ("type", "upload_date", "sort_by", "duration", "cursor"),
            True,
            (
                ("type", ("videos", "shorts", "channels", "playlists")),
                ("upload_date", ("today", "this_week", "this_month", "this_year")),
                ("sort_by", ("relevance", "popular")),
                (
                    "duration",
                    ("under_3_min", "between_3_and_20_min", "over_20_min"),
                ),
            ),
        ),
        "video": SocialAction("youtube_video", ("id",)),
        "transcript": SocialAction("youtube_transcript", ("video_id",), ("language",)),
        "comments": SocialAction(
            "youtube_comments",
            ("video_id",),
            ("order", "cursor"),
            True,
            (("order", ("top", "newest")),),
        ),
        "channel": SocialAction("youtube_channel", ("handle",)),
        "channel_videos": SocialAction(
            "youtube_channel_videos",
            ("handle",),
            ("sort", "cursor"),
            True,
            (("sort", ("latest", "popular")),),
        ),
    },
    "reddit": {
        "search": SocialAction(
            "reddit_search",
            ("query",),
            ("sort", "timeframe", "cursor"),
            True,
            (
                ("sort", ("relevance", "new", "top", "comment_count")),
                ("timeframe", _TIMEFRAMES),
            ),
        ),
        "subreddit": SocialAction("reddit_subreddit", ("name",)),
        "subreddit_posts": SocialAction(
            "reddit_subreddit_posts",
            ("name",),
            ("sort", "timeframe", "cursor"),
            True,
            (
                ("sort", ("best", "hot", "new", "top", "rising")),
                ("timeframe", _TIMEFRAMES),
            ),
        ),
        "post": SocialAction("reddit_post_by_id", ("id",)),
        "comments": SocialAction("reddit_comments", ("post_id",), ("cursor",), True),
    },
    "threads": {
        "search": SocialAction(
            "threads_search", ("query",), ("start_date", "end_date"), True
        ),
        "search_users": SocialAction("threads_search_users", ("query",), (), True),
        "profile": SocialAction("threads_profile", ("username",)),
        "profile_posts": SocialAction(
            "threads_profile_posts", ("username",), (), True
        ),
        "post": SocialAction("threads_post_by_id", ("id",)),
    },
}

_REGION = re.compile(r"^[A-Z]{2}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Leading characters people often type that the API does not want. A YouTube
# handle keeps its @: the API takes it with or without.
_STRIP_PREFIX = {"username": "@", "profile_id": "@", "tag": "#", "name": "r/"}


@dataclass(frozen=True)
class SocialRequest:
    """A validated call: the SDK method, its keyword arguments and the response kind."""

    method: str
    kwargs: Dict[str, Any]
    returns_list: bool


def _clean(name: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    prefix = _STRIP_PREFIX.get(name)
    if prefix and value.startswith(prefix):
        value = value[len(prefix) :].strip()
    if name == "region":
        value = value.upper()
    return value


def _describe(action: str, spec: SocialAction) -> str:
    parts = [f"{name} (required)" for name in spec.required] + list(spec.optional)
    return f"Fields for '{action}': {', '.join(parts) if parts else 'none'}."


def social_request(platform: str, action: str, **fields: Any) -> SocialRequest:
    """Check an action's fields and build its SDK call.

    Raises ValueError with a message that says what the action needs, before
    any request is sent. Empty strings count as unset. ``cursor`` is passed
    through unchanged.
    """
    actions = SOCIAL_ACTIONS.get(platform)
    if actions is None:
        raise ValueError(
            f"Unknown platform {platform!r}. Use one of: {', '.join(SOCIAL_PLATFORMS)}."
        )
    spec = actions.get(action)
    if spec is None:
        raise ValueError(
            f"Unknown {platform} action {action!r}. Use one of: {', '.join(actions)}."
        )

    # Empty strings count as unset: some models fill every field of a tool.
    values: Dict[str, Any] = {}
    for name, value in fields.items():
        if name == "cursor":
            # Opaque: sent back exactly as received, never cleaned.
            if isinstance(value, str) and not value.strip():
                value = None
        else:
            value = _clean(name, value)
        if value is not None and value != "":
            values[name] = value

    extra = [name for name in values if name not in spec.fields]
    if extra:
        verb = "does" if len(extra) == 1 else "do"
        message = (
            f"{', '.join(extra)} {verb} not apply to {platform} action '{action}'. "
            f"{_describe(action, spec)}"
        )
        if "cursor" in extra and spec.returns_list:
            message += f" '{action}' returns one page and does not paginate."
        raise ValueError(message)

    missing = [name for name in spec.required if name not in values]
    if missing:
        raise ValueError(
            f"{platform} action '{action}' needs {', '.join(missing)}. "
            f"{_describe(action, spec)}"
        )

    region = values.get("region")
    if region is not None and not _REGION.match(str(region)):
        raise ValueError(
            f"region must be a two-letter code such as US, got {region!r}."
        )

    for name, allowed in spec.choices:
        value = values.get(name)
        if value is not None and value not in allowed:
            raise ValueError(
                f"{name} for {platform} action '{action}' must be one of "
                f"{', '.join(allowed)}; got {value!r}."
            )

    for name in ("start_date", "end_date"):
        value = values.get(name)
        if value is not None and not _DATE.match(str(value)):
            raise ValueError(f"{name} must be a date as YYYY-MM-DD, got {value!r}.")

    return SocialRequest(spec.method, values, spec.returns_list)


def social_page(result: Any, tool_name: str) -> Any:
    """Turn ``DesearchResponse`` output into ``{"results", "next_cursor"}``.

    Error strings from the tool layer are returned unchanged.
    """
    if not isinstance(result, dict) or "data" not in result:
        return result
    metadata = result.get("metadata")
    next_cursor: Optional[str] = None
    if isinstance(metadata, dict):
        next_cursor = metadata.get("next_cursor") or None
    page: Dict[str, Any] = {"results": result["data"], "next_cursor": next_cursor}
    if next_cursor:
        page["next_step"] = (
            f"More results exist. Call {tool_name} again with the same action and "
            "target and cursor set to next_cursor, unchanged."
        )
    return page
