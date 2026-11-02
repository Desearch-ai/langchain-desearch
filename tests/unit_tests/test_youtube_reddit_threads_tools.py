"""YouTube, Reddit and Threads tools (1.3.0), with the SDK client mocked.

The fake client checks every call against the signature of the installed
desearch-py method of the same name, so a wrong method name or keyword fails
the test instead of passing silently.
"""

import asyncio
import inspect

import pytest
from desearch_py import (
    Desearch,
    DesearchAPIError,
    DesearchCostMetadata,
    DesearchResponse,
    RedditComment,
    RedditPost,
    RedditSubreddit,
    ThreadsPost,
    ThreadsPostDetail,
    ThreadsProfile,
    ThreadsUser,
    YouTubeChannel,
    YouTubeComment,
    YouTubeItem,
    YouTubeTranscript,
    YouTubeVideo,
)
from pydantic import ValidationError

from langchain_desearch.search_tools import social_tools
from langchain_desearch.social import SOCIAL_ACTIONS, social_request
from langchain_desearch.tools import RedditTool, ThreadsTool, YouTubeTool

CURSOR = "eyJwIjoieW91dHViZSJ9=+/opaque"
NEXT_CURSOR = "bmV4dA==/next+page"


def _payloads():
    """Return values shaped like desearch-py 1.11.0's, model instances included."""
    return {
        "youtube_search": [YouTubeItem(id="dQw4w9WgXcQ", type="video", title="t")],
        "youtube_video": YouTubeVideo(id="dQw4w9WgXcQ", title="t", views=1),
        "youtube_transcript": YouTubeTranscript(
            videoId="dQw4w9WgXcQ", text="hi", segments=[{"text": "hi", "startMs": 0}]
        ),
        "youtube_comments": [YouTubeComment(id="c1", text="nice")],
        "youtube_channel": YouTubeChannel(handle="MrBeast", subscribers=10),
        "youtube_channel_videos": [YouTubeItem(id="v2", type="video")],
        "reddit_search": [RedditPost(id="1q6pxwn", title="t")],
        "reddit_subreddit": RedditSubreddit(name="AskReddit"),
        "reddit_subreddit_posts": [RedditPost(id="p2")],
        "reddit_post_by_id": RedditPost(id="1q6pxwn", title="t"),
        "reddit_comments": [
            RedditComment(id="c1", replies=[{"id": "c2", "depth": 1}], depth=0)
        ],
        "threads_search": [ThreadsPost(id="1", code="DK-BydcJHkF")],
        "threads_search_users": [ThreadsUser(username="zuck")],
        "threads_profile": ThreadsProfile(username="zuck", followers=10),
        "threads_profile_posts": [ThreadsPost(id="2", code="X")],
        "threads_post_by_id": ThreadsPostDetail(
            id="1", code="DK-BydcJHkF", comments=[{"id": "r1"}]
        ),
    }


class _FakeDesearch:
    created = []
    next_cursor = None
    error = None

    def __init__(self, api_key: str, base_url: str = "https://api.desearch.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []
        self.closed = False
        _FakeDesearch.created.append(self)

    def __getattr__(self, name):
        real = getattr(Desearch, name)  # AttributeError for a method the SDK lacks

        async def method(**kwargs):
            inspect.signature(real).bind(self, **kwargs)
            self.calls.append((name, kwargs))
            if _FakeDesearch.error is not None:
                raise _FakeDesearch.error
            data = _payloads()[name]
            if kwargs.get("include_metadata"):
                return DesearchResponse(
                    data=data,
                    metadata=DesearchCostMetadata(
                        cost_usd=0.00207, next_cursor=_FakeDesearch.next_cursor
                    ),
                )
            return data

        return method

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_desearch(monkeypatch):
    _FakeDesearch.created = []
    _FakeDesearch.next_cursor = None
    _FakeDesearch.error = None
    monkeypatch.setenv("DESEARCH_API_KEY", "test-key")
    monkeypatch.setattr("langchain_desearch.tools.Desearch", _FakeDesearch)
    return _FakeDesearch


def _last_call(fake):
    client = fake.created[-1]
    assert client.closed is True
    assert client.api_key == "test-key"
    return client.calls[-1]


# (tool, payload, SDK method, SDK kwargs without include_metadata, list?)
CASES = [
    (
        YouTubeTool,
        {
            "action": "search",
            "query": "bittensor",
            "type": "videos",
            "upload_date": "this_week",
            "sort_by": "popular",
            "duration": "over_20_min",
        },
        "youtube_search",
        {
            "query": "bittensor",
            "type": "videos",
            "upload_date": "this_week",
            "sort_by": "popular",
            "duration": "over_20_min",
        },
        True,
    ),
    (YouTubeTool, {"action": "video", "id": "dQw4w9WgXcQ"}, "youtube_video", {"id": "dQw4w9WgXcQ"}, False),
    (
        YouTubeTool,
        {"action": "transcript", "video_id": "dQw4w9WgXcQ", "language": "en"},
        "youtube_transcript",
        {"video_id": "dQw4w9WgXcQ", "language": "en"},
        False,
    ),
    (
        YouTubeTool,
        {"action": "comments", "video_id": "dQw4w9WgXcQ", "order": "newest"},
        "youtube_comments",
        {"video_id": "dQw4w9WgXcQ", "order": "newest"},
        True,
    ),
    (
        YouTubeTool,
        {"action": "channel", "handle": "@MrBeast"},
        "youtube_channel",
        {"handle": "@MrBeast"},
        False,
    ),
    (
        YouTubeTool,
        {"action": "channel_videos", "handle": "@MrBeast", "sort": "popular"},
        "youtube_channel_videos",
        {"handle": "@MrBeast", "sort": "popular"},
        True,
    ),
    (
        RedditTool,
        {"action": "search", "query": "bittensor", "sort": "comment_count", "timeframe": "month"},
        "reddit_search",
        {"query": "bittensor", "sort": "comment_count", "timeframe": "month"},
        True,
    ),
    (RedditTool, {"action": "subreddit", "name": "AskReddit"}, "reddit_subreddit", {"name": "AskReddit"}, False),
    (
        RedditTool,
        {"action": "subreddit_posts", "name": "AskReddit", "sort": "top", "timeframe": "week"},
        "reddit_subreddit_posts",
        {"name": "AskReddit", "sort": "top", "timeframe": "week"},
        True,
    ),
    (RedditTool, {"action": "post", "id": "1q6pxwn"}, "reddit_post_by_id", {"id": "1q6pxwn"}, False),
    (RedditTool, {"action": "comments", "post_id": "1q6pxwn"}, "reddit_comments", {"post_id": "1q6pxwn"}, True),
    (
        ThreadsTool,
        {"action": "search", "query": "bittensor", "start_date": "2026-01-01", "end_date": "2026-10-30"},
        "threads_search",
        {"query": "bittensor", "start_date": "2026-01-01", "end_date": "2026-10-30"},
        True,
    ),
    (ThreadsTool, {"action": "search_users", "query": "zuck"}, "threads_search_users", {"query": "zuck"}, True),
    (ThreadsTool, {"action": "profile", "username": "zuck"}, "threads_profile", {"username": "zuck"}, False),
    (ThreadsTool, {"action": "profile_posts", "username": "zuck"}, "threads_profile_posts", {"username": "zuck"}, True),
    (ThreadsTool, {"action": "post", "id": "DK-BydcJHkF"}, "threads_post_by_id", {"id": "DK-BydcJHkF"}, False),
]


def test_tool_names_exports_and_group():
    import langchain_desearch

    for tool_cls, name in (
        (YouTubeTool, "youtube_tool"),
        (RedditTool, "reddit_tool"),
        (ThreadsTool, "threads_tool"),
    ):
        tool = tool_cls()
        assert tool.name == name
        assert tool.return_direct is True
        assert tool.description
        assert getattr(langchain_desearch, tool_cls.__name__) is tool_cls
        assert name in [t.name for t in social_tools]


def test_actions_cover_all_sixteen_endpoints():
    methods = {
        spec.method
        for platform in ("youtube", "reddit", "threads")
        for spec in SOCIAL_ACTIONS[platform].values()
    }
    assert methods == {case[2] for case in CASES}
    assert len(methods) == 16
    paged = {
        spec.method
        for platform in ("youtube", "reddit", "threads")
        for spec in SOCIAL_ACTIONS[platform].values()
        if spec.paginates
    }
    assert paged == {
        "youtube_search",
        "youtube_comments",
        "youtube_channel_videos",
        "reddit_search",
        "reddit_subreddit_posts",
        "reddit_comments",
    }
    for method in methods:
        parameters = inspect.signature(getattr(Desearch, method)).parameters
        assert "include_metadata" in parameters
        assert ("cursor" in parameters) == (method in paged), method
        assert "count" not in parameters


@pytest.mark.parametrize("tool_cls,payload,method,expected,is_list", CASES)
def test_each_action_calls_the_sdk_method(
    fake_desearch, tool_cls, payload, method, expected, is_list
):
    result = tool_cls().invoke(payload)
    name, kwargs = _last_call(fake_desearch)
    assert name == method
    if is_list:
        assert kwargs == {**expected, "include_metadata": True}
        assert set(result) == {"results", "next_cursor"}
        assert result["next_cursor"] is None
        assert isinstance(result["results"], list) and result["results"]
        assert isinstance(result["results"][0], dict)
    else:
        assert kwargs == expected
        assert isinstance(result, dict)


@pytest.mark.parametrize("tool_cls,payload,method,expected,is_list", CASES)
def test_each_action_async_path(
    fake_desearch, tool_cls, payload, method, expected, is_list
):
    result = asyncio.run(tool_cls().ainvoke(payload))
    name, kwargs = _last_call(fake_desearch)
    assert name == method
    assert isinstance(result, dict)


def test_results_are_plain_json(fake_desearch):
    tree = RedditTool().invoke({"action": "comments", "post_id": "1q6pxwn"})
    assert tree["results"][0]["replies"][0]["id"] == "c2"
    post = ThreadsTool().invoke({"action": "post", "id": "DK-BydcJHkF"})
    assert post["comments"][0]["id"] == "r1"
    assert post["threadItems"] == []
    transcript = YouTubeTool().invoke(
        {"action": "transcript", "video_id": "dQw4w9WgXcQ"}
    )
    assert transcript["segments"][0]["text"] == "hi"


@pytest.mark.parametrize(
    "tool_cls,payload,method",
    [
        (YouTubeTool, {"action": "search", "query": "ai"}, "youtube_search"),
        (YouTubeTool, {"action": "comments", "video_id": "v"}, "youtube_comments"),
        (YouTubeTool, {"action": "channel_videos", "handle": "h"}, "youtube_channel_videos"),
        (RedditTool, {"action": "search", "query": "ai"}, "reddit_search"),
        (RedditTool, {"action": "subreddit_posts", "name": "AskReddit"}, "reddit_subreddit_posts"),
        (RedditTool, {"action": "comments", "post_id": "p"}, "reddit_comments"),
    ],
)
def test_cursor_passthrough(fake_desearch, tool_cls, payload, method):
    fake_desearch.next_cursor = NEXT_CURSOR
    page = tool_cls().invoke({**payload, "cursor": CURSOR})
    name, kwargs = _last_call(fake_desearch)
    assert name == method
    assert kwargs["cursor"] == CURSOR
    assert page["next_cursor"] == NEXT_CURSOR
    assert "cursor set to next_cursor" in page["next_step"]


def test_targets_are_cleaned(fake_desearch):
    RedditTool().invoke({"action": "subreddit", "name": " r/AskReddit "})
    assert _last_call(fake_desearch)[1] == {"name": "AskReddit"}
    ThreadsTool().invoke({"action": "profile", "username": "@zuck"})
    assert _last_call(fake_desearch)[1] == {"username": "zuck"}
    # A YouTube handle keeps its @: the API takes it with or without.
    YouTubeTool().invoke({"action": "channel", "handle": "@MrBeast"})
    assert _last_call(fake_desearch)[1] == {"handle": "@MrBeast"}


def test_empty_strings_count_as_unset(fake_desearch):
    YouTubeTool().invoke(
        {"action": "search", "query": "ai", "sort_by": None, "cursor": "  ", "id": ""}
    )
    assert _last_call(fake_desearch)[1] == {"query": "ai", "include_metadata": True}


@pytest.mark.parametrize(
    "tool_cls,payload,message",
    [
        (YouTubeTool, {"action": "video"}, "needs id"),
        (YouTubeTool, {"action": "transcript", "id": "x"}, "id does not apply"),
        (YouTubeTool, {"action": "video", "id": "x", "cursor": CURSOR}, "cursor does not apply"),
        (YouTubeTool, {"action": "channel", "handle": "h", "sort": "latest"}, "sort does not apply"),
        (YouTubeTool, {"action": "search", "query": "ai", "sort": "popular"}, "sort does not apply"),
        (YouTubeTool, {"action": "search", "query": "ai", "type": "music"}, "type"),
        (RedditTool, {"action": "search", "query": "ai", "sort": "hot"}, "relevance, new, top, comment_count"),
        (RedditTool, {"action": "subreddit_posts", "name": "x", "sort": "relevance"}, "best, hot, new, top, rising"),
        (RedditTool, {"action": "post", "post_id": "x"}, "post_id does not apply"),
        (ThreadsTool, {"action": "search", "query": "ai", "start_date": "01/02/2026"}, "YYYY-MM-DD"),
        (ThreadsTool, {"action": "profile_posts", "username": "zuck", "query": "x"}, "query does not apply"),
    ],
)
def test_inputs_are_validated_before_any_call(fake_desearch, tool_cls, payload, message):
    with pytest.raises(ValidationError, match=message):
        tool_cls().invoke(payload)
    assert fake_desearch.created == []


def test_threads_takes_no_cursor():
    assert "cursor" not in ThreadsTool().args
    with pytest.raises(ValueError, match="does not paginate"):
        social_request("threads", "search", query="ai", cursor=CURSOR)


def test_youtube_and_reddit_take_no_count():
    assert "count" not in YouTubeTool().args
    assert "count" not in RedditTool().args
    with pytest.raises(ValueError, match="count does not apply"):
        social_request("reddit", "search", query="ai", count=5)


def test_sdk_errors_are_returned_as_messages(fake_desearch):
    fake_desearch.error = DesearchAPIError(404, "Not Found", '{"detail":"Not found"}')
    result = YouTubeTool().invoke({"action": "transcript", "video_id": "nocaptions"})
    assert isinstance(result, str)
    assert result.startswith("An error occurred while calling Desearch")
    assert "404" in result
