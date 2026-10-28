"""Tools added for full Desearch API coverage (1.2.0), with the SDK client mocked.

The fake client checks every call against the signature of the installed
desearch-py method of the same name, so a wrong method name or keyword fails
the test instead of passing silently.
"""

import asyncio
import inspect

import pytest
from desearch_py import (
    Desearch,
    DesearchCostMetadata,
    DesearchResponse,
    FacebookComment,
    FacebookPost,
    FacebookProfile,
    InstagramComment,
    InstagramMedia,
    InstagramProfile,
    InstagramSearchResult,
    SearchApiResponse,
    TikTokComment,
    TikTokPost,
    TikTokProfile,
    TwitterScraperTweet,
    WebSearchResponse,
    XLinksSearchResponse,
    XTrendsResponse,
)
from pydantic import ValidationError

from langchain_desearch.search_tools import (
    people_tools,
    search_tools,
    social_tools,
    twitter_tools,
)
from langchain_desearch.social import SOCIAL_ACTIONS, social_request
from langchain_desearch.tools import (
    AIWebLinksSearchTool,
    AIXLinksSearchTool,
    FacebookTool,
    FetchTwitterTrendsTool,
    InstagramTool,
    SearchApiTool,
    SearchTweetsByUserTool,
    TikTokTool,
    WebExtractTool,
)

CURSOR = "eyJwIjoidGlrdG9rIn0=+/opaque"
NEXT_CURSOR = "bmV4dA==/next+page"


def _tweet(tweet_id="1"):
    return {
        "user": {"id": "u1", "username": "elonmusk", "name": "Elon"},
        "id": tweet_id,
        "text": "post",
        "reply_count": 0,
        "retweet_count": 0,
        "like_count": 0,
        "quote_count": 0,
        "bookmark_count": 0,
        "url": f"https://x.com/elonmusk/status/{tweet_id}",
        "created_at": "2026-10-01T00:00:00Z",
    }


def _payloads():
    """Return values shaped like desearch-py 1.8.0's, model instances included."""
    return {
        "ai_web_links_search": WebSearchResponse(
            search_results=[
                {
                    "title": "Desearch",
                    "snippet": "AI search",
                    "link": "https://desearch.ai",
                }
            ]
        ),
        "ai_x_links_search": XLinksSearchResponse(miner_tweets=[_tweet()]),
        "x_posts_by_user": [TwitterScraperTweet(**_tweet())],
        "x_trends": XTrendsResponse(
            trends=[{"name": "#AI", "rank": 1}], woeid={"name": "Worldwide", "id": 1}
        ),
        "extract": "Desearch page text",
        "search_api": SearchApiResponse(
            results=[
                {
                    "url": "https://desearch.ai",
                    "title": "Desearch",
                    "highlights": ["AI search"],
                }
            ],
            cost_usd=0.002,
        ),
        "tiktok_search": [TikTokPost(id="7300000000000000001", text="ai")],
        "tiktok_post_by_id": TikTokPost(id="7300000000000000001", text="ai"),
        "tiktok_profile": TikTokProfile(username="tiktok", followers=10),
        "tiktok_profile_posts": [TikTokPost(id="7300000000000000002")],
        "tiktok_comments": [TikTokComment(id="c1", text="nice")],
        "tiktok_hashtag": [TikTokPost(id="7300000000000000003")],
        "tiktok_trending": [TikTokPost(id="7300000000000000004")],
        "facebook_search": [FacebookPost(id="ad1", headline="Ad")],
        "facebook_post_by_id": FacebookPost(id="p1", text="hello"),
        "facebook_profile": FacebookProfile(id="20531316728", username="facebook"),
        "facebook_profile_posts": [FacebookPost(id="p2")],
        "facebook_comments": [FacebookComment(id="fc1", text="hi")],
        "facebook_hashtag": [FacebookPost(id="ad2")],
        "instagram_search": [InstagramSearchResult(type="user", username="instagram")],
        "instagram_media": InstagramMedia(id="m1", shortcode="DY79rTVS6OK"),
        "instagram_profile": InstagramProfile(username="instagram"),
        "instagram_profile_posts": [InstagramMedia(id="m2", shortcode="ABC")],
        "instagram_comments": [InstagramComment(id="ic1", text="wow")],
        "instagram_hashtag": [InstagramMedia(id="m3")],
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
                        cost_usd=0.001, next_cursor=_FakeDesearch.next_cursor
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


# ─── Names, exports and groups ───────────────────────────────────────────────

NEW_TOOLS = {
    AIWebLinksSearchTool: "ai_web_links_search_tool",
    AIXLinksSearchTool: "ai_x_links_search_tool",
    SearchTweetsByUserTool: "search_tweets_by_user_tool",
    FetchTwitterTrendsTool: "fetch_twitter_trends_tool",
    WebExtractTool: "web_extract_tool",
    SearchApiTool: "search_api_tool",
    TikTokTool: "tiktok_tool",
    FacebookTool: "facebook_tool",
    InstagramTool: "instagram_tool",
}


def test_new_tool_names_and_exports():
    import langchain_desearch

    for tool_cls, name in NEW_TOOLS.items():
        tool = tool_cls()
        assert tool.name == name
        assert tool.return_direct is True
        assert tool.description
        assert tool_cls.__name__ in langchain_desearch.__all__
        assert getattr(langchain_desearch, tool_cls.__name__) is tool_cls


def test_new_tools_are_in_the_tool_groups():
    assert [tool.name for tool in search_tools] == [
        "desearch_tool",
        "basic_web_search_tool",
        "basic_twitter_search_tool",
        "ai_web_links_search_tool",
        "ai_x_links_search_tool",
        "search_api_tool",
        "web_extract_tool",
    ]
    twitter_names = [tool.name for tool in twitter_tools]
    for name in (
        "search_tweets_by_user_tool",
        "fetch_twitter_trends_tool",
        "ai_x_links_search_tool",
    ):
        assert name in twitter_names
    assert [tool.name for tool in social_tools] == [
        "tiktok_tool",
        "facebook_tool",
        "instagram_tool",
    ]
    # The billed People Search job never joins the general lists.
    for group in (search_tools, twitter_tools, social_tools):
        assert "people_search_tool" not in [tool.name for tool in group]
    assert [tool.name for tool in people_tools] == [
        "people_search_tool",
        "people_profiles_tool",
    ]


def test_social_actions_cover_all_nineteen_endpoints():
    methods = {
        spec.method for actions in SOCIAL_ACTIONS.values() for spec in actions.values()
    }
    assert len(methods) == 19
    for method in methods:
        assert inspect.iscoroutinefunction(getattr(Desearch, method)), method
    for actions in SOCIAL_ACTIONS.values():
        for action, spec in actions.items():
            # Every list route gets include_metadata so the cursor comes back.
            assert (
                "include_metadata"
                in inspect.signature(getattr(Desearch, spec.method)).parameters
            )
            if spec.paginates:
                assert (
                    "cursor"
                    in inspect.signature(getattr(Desearch, spec.method)).parameters
                ), action


# ─── X, AI links, extract and Search API tools ───────────────────────────────

SIMPLE_CASES = [
    (
        AIWebLinksSearchTool,
        {"prompt": "Bittensor subnets", "count": 10},
        "ai_web_links_search",
        {"prompt": "Bittensor subnets", "tools": ["web"], "count": 10},
    ),
    (
        AIWebLinksSearchTool,
        {"prompt": "Bittensor subnets"},
        "ai_web_links_search",
        {"prompt": "Bittensor subnets", "tools": ["web"]},
    ),
    (
        AIXLinksSearchTool,
        {"prompt": "Bittensor news", "count": 25},
        "ai_x_links_search",
        {"prompt": "Bittensor news", "count": 25},
    ),
    (
        SearchTweetsByUserTool,
        {"user": "elonmusk", "query": "tesla", "count": 3},
        "x_posts_by_user",
        {"user": "elonmusk", "query": "tesla", "count": 3},
    ),
    (
        SearchTweetsByUserTool,
        {"user": "elonmusk"},
        "x_posts_by_user",
        {"user": "elonmusk"},
    ),
    (FetchTwitterTrendsTool, {}, "x_trends", {"woeid": 1}),
    (
        FetchTwitterTrendsTool,
        {"woeid": 23424977, "count": 30},
        "x_trends",
        {"woeid": 23424977, "count": 30},
    ),
    (
        WebExtractTool,
        {"url": "https://desearch.ai"},
        "extract",
        {"url": "https://desearch.ai", "format": "text", "js": False},
    ),
    (
        WebExtractTool,
        {"url": "https://desearch.ai", "format": "html", "js": True, "wait": 250},
        "extract",
        {"url": "https://desearch.ai", "format": "html", "js": True, "wait": 250},
    ),
    (
        SearchApiTool,
        {"query": "Nobel Prize in Physics 2024"},
        "search_api",
        {"query": "Nobel Prize in Physics 2024"},
    ),
    (
        SearchApiTool,
        {
            "query": "Nobel Prize in Physics 2024",
            "mode": "fast",
            "count": 15,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "highlights": False,
            "page_text": True,
        },
        "search_api",
        {
            "query": "Nobel Prize in Physics 2024",
            "mode": "fast",
            "count": 15,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "highlights": False,
            "page_text": True,
        },
    ),
]


@pytest.mark.parametrize(("tool_cls", "payload", "method", "expected"), SIMPLE_CASES)
def test_new_tools_call_the_sdk_method_with_kwargs(
    fake_desearch, tool_cls, payload, method, expected
):
    result = tool_cls().invoke(payload)

    called, kwargs = _last_call(fake_desearch)
    assert called == method
    assert kwargs == expected
    assert result is not None
    assert not (isinstance(result, str) and result.startswith("An error occurred"))


@pytest.mark.parametrize(("tool_cls", "payload", "method", "expected"), SIMPLE_CASES)
def test_new_tools_async_path(fake_desearch, tool_cls, payload, method, expected):
    sync_result = tool_cls().invoke(payload)
    async_result = asyncio.run(tool_cls().ainvoke(payload))

    called, kwargs = _last_call(fake_desearch)
    assert called == method
    assert kwargs == expected
    assert async_result == sync_result


def test_new_tools_return_plain_json(fake_desearch):
    links = AIWebLinksSearchTool().invoke({"prompt": "Desearch"})
    assert links["search_results"] == [
        {"title": "Desearch", "snippet": "AI search", "link": "https://desearch.ai"}
    ]
    tweets = AIXLinksSearchTool().invoke({"prompt": "Desearch"})
    assert tweets["miner_tweets"][0]["id"] == "1"
    posts = SearchTweetsByUserTool().invoke({"user": "elonmusk"})
    assert isinstance(posts, list) and posts[0]["id"] == "1"
    trends = FetchTwitterTrendsTool().invoke({})
    assert trends["trends"][0]["name"] == "#AI"
    assert (
        WebExtractTool().invoke({"url": "https://desearch.ai"}) == "Desearch page text"
    )
    search = SearchApiTool().invoke({"query": "Desearch"})
    assert search["results"][0]["highlights"] == ["AI search"]
    assert search["cost_usd"] == 0.002


@pytest.mark.parametrize(
    ("tool_cls", "payload", "message"),
    [
        (
            AIWebLinksSearchTool,
            {"prompt": "x", "count": 5},
            "greater than or equal to 10",
        ),
        (
            AIWebLinksSearchTool,
            {"prompt": "x", "count": 201},
            "less than or equal to 200",
        ),
        (AIWebLinksSearchTool, {"prompt": ""}, "at least 1 character"),
        (
            AIXLinksSearchTool,
            {"prompt": "x", "count": 9},
            "greater than or equal to 10",
        ),
        (
            SearchTweetsByUserTool,
            {"user": "elonmusk", "count": 0},
            "greater than or equal to 1",
        ),
        (
            SearchTweetsByUserTool,
            {"user": "elonmusk", "count": 101},
            "less than or equal to 100",
        ),
        (SearchTweetsByUserTool, {"query": "tesla"}, "user"),
        (FetchTwitterTrendsTool, {"count": 10}, "greater than or equal to 30"),
        (FetchTwitterTrendsTool, {"count": 101}, "less than or equal to 100"),
        (
            WebExtractTool,
            {"url": "https://desearch.ai", "format": "pdf"},
            "'text' or 'html'",
        ),
        (
            WebExtractTool,
            {"url": "https://desearch.ai", "wait": 100},
            "wait only applies when js is true",
        ),
        (
            WebExtractTool,
            {"url": "https://desearch.ai", "js": True, "wait": 30001},
            "less than or equal to 30000",
        ),
        (SearchApiTool, {"query": "x", "count": 12}, "10, 15 or 20"),
        (SearchApiTool, {"query": "x", "mode": "deep"}, "'fast' or 'balanced'"),
        (SearchApiTool, {"query": ""}, "at least 1 character"),
        (SearchApiTool, {"query": "x" * 2001}, "at most 2000 characters"),
        (SearchApiTool, {"query": "x", "start_date": "01/02/2024"}, "YYYY-MM-DD"),
        (SearchApiTool, {"query": "x", "end_date": "2024-02-30"}, "not a valid date"),
        (
            SearchApiTool,
            {"query": "x", "start_date": "2024-12-31", "end_date": "2024-01-01"},
            "start_date must be on or before end_date",
        ),
    ],
)
def test_new_tool_inputs_are_validated_before_any_call(
    fake_desearch, tool_cls, payload, message
):
    with pytest.raises(ValidationError, match=message):
        tool_cls().invoke(payload)
    assert fake_desearch.created == []


# ─── TikTok, Facebook and Instagram ──────────────────────────────────────────

SOCIAL_CASES = [
    # TikTok
    (
        TikTokTool,
        {"action": "search", "query": "ai", "count": 3, "sort": "latest"},
        "tiktok_search",
        {"query": "ai", "count": 3, "sort": "latest"},
        True,
    ),
    (
        TikTokTool,
        {"action": "post", "id": "7300000000000000001"},
        "tiktok_post_by_id",
        {"id": "7300000000000000001"},
        False,
    ),
    (
        TikTokTool,
        {"action": "profile", "username": "tiktok"},
        "tiktok_profile",
        {"username": "tiktok"},
        False,
    ),
    (
        TikTokTool,
        {"action": "profile_posts", "username": "tiktok", "count": 2},
        "tiktok_profile_posts",
        {"username": "tiktok", "count": 2},
        True,
    ),
    (
        TikTokTool,
        {"action": "comments", "video_id": "7300000000000000001", "count": 1},
        "tiktok_comments",
        {"video_id": "7300000000000000001", "count": 1},
        True,
    ),
    (
        TikTokTool,
        {"action": "hashtag", "tag": "ai"},
        "tiktok_hashtag",
        {"tag": "ai"},
        True,
    ),
    (
        TikTokTool,
        {"action": "trending", "region": "gb", "count": 3},
        "tiktok_trending",
        {"region": "GB", "count": 3},
        True,
    ),
    (TikTokTool, {"action": "trending"}, "tiktok_trending", {}, True),
    # Facebook
    (
        FacebookTool,
        {"action": "search", "query": "running shoes", "count": 2},
        "facebook_search",
        {"query": "running shoes", "count": 2},
        True,
    ),
    (
        FacebookTool,
        {"action": "post", "id": "10153231379946729"},
        "facebook_post_by_id",
        {"id": "10153231379946729"},
        False,
    ),
    (
        FacebookTool,
        {"action": "profile", "profile_id": "facebook"},
        "facebook_profile",
        {"profile_id": "facebook"},
        False,
    ),
    (
        FacebookTool,
        {"action": "profile_posts", "profile_id": "facebook", "count": 3},
        "facebook_profile_posts",
        {"profile_id": "facebook", "count": 3},
        True,
    ),
    (
        FacebookTool,
        {"action": "comments", "post_id": "10153231379946729"},
        "facebook_comments",
        {"post_id": "10153231379946729"},
        True,
    ),
    (
        FacebookTool,
        {"action": "hashtag", "tag": "ai", "count": 1},
        "facebook_hashtag",
        {"tag": "ai", "count": 1},
        True,
    ),
    # Instagram
    (
        InstagramTool,
        {"action": "search", "query": "instagram", "count": 3},
        "instagram_search",
        {"query": "instagram", "count": 3},
        True,
    ),
    (
        InstagramTool,
        {"action": "post", "id": "https://www.instagram.com/p/DY79rTVS6OK/"},
        "instagram_media",
        {"id": "https://www.instagram.com/p/DY79rTVS6OK/"},
        False,
    ),
    (
        InstagramTool,
        {"action": "profile", "username": "instagram"},
        "instagram_profile",
        {"username": "instagram"},
        False,
    ),
    (
        InstagramTool,
        {"action": "profile_posts", "username": "instagram", "count": 2},
        "instagram_profile_posts",
        {"username": "instagram", "count": 2},
        True,
    ),
    (
        InstagramTool,
        {"action": "comments", "shortcode": "DeuOR-CjKE9"},
        "instagram_comments",
        {"shortcode": "DeuOR-CjKE9"},
        True,
    ),
    (
        InstagramTool,
        {"action": "hashtag", "tag": "ai", "count": 1},
        "instagram_hashtag",
        {"tag": "ai", "count": 1},
        True,
    ),
]


@pytest.mark.parametrize(
    ("tool_cls", "payload", "method", "expected", "is_list"), SOCIAL_CASES
)
def test_social_actions_call_the_sdk_method_with_kwargs(
    fake_desearch, tool_cls, payload, method, expected, is_list
):
    result = tool_cls().invoke(payload)

    called, kwargs = _last_call(fake_desearch)
    assert called == method
    if is_list:
        assert kwargs == {**expected, "include_metadata": True}
        assert set(result) == {"results", "next_cursor"}
        assert isinstance(result["results"], list)
        assert isinstance(result["results"][0], dict)
        assert result["next_cursor"] is None
    else:
        assert kwargs == expected
        assert isinstance(result, dict)
        assert "next_cursor" not in result


@pytest.mark.parametrize(
    ("tool_cls", "payload", "method", "expected", "is_list"), SOCIAL_CASES
)
def test_social_actions_async_path(
    fake_desearch, tool_cls, payload, method, expected, is_list
):
    _FakeDesearch.next_cursor = NEXT_CURSOR if is_list else None
    sync_result = tool_cls().invoke(payload)
    async_result = asyncio.run(tool_cls().ainvoke(payload))

    called, kwargs = _last_call(fake_desearch)
    assert called == method
    assert kwargs == ({**expected, "include_metadata": True} if is_list else expected)
    assert async_result == sync_result


def test_social_result_is_plain_json(fake_desearch):
    page = TikTokTool().invoke({"action": "hashtag", "tag": "ai"})
    assert page["results"][0]["id"] == "7300000000000000003"
    profile = InstagramTool().invoke({"action": "profile", "username": "instagram"})
    assert profile["username"] == "instagram"


@pytest.mark.parametrize(
    ("tool_cls", "payload", "method"),
    [
        (TikTokTool, {"action": "search", "query": "ai"}, "tiktok_search"),
        (
            TikTokTool,
            {"action": "profile_posts", "username": "tiktok"},
            "tiktok_profile_posts",
        ),
        (TikTokTool, {"action": "comments", "video_id": "1"}, "tiktok_comments"),
        (TikTokTool, {"action": "hashtag", "tag": "ai"}, "tiktok_hashtag"),
        (FacebookTool, {"action": "search", "query": "ai"}, "facebook_search"),
        (
            FacebookTool,
            {"action": "profile_posts", "profile_id": "facebook"},
            "facebook_profile_posts",
        ),
        (FacebookTool, {"action": "comments", "post_id": "1"}, "facebook_comments"),
        (FacebookTool, {"action": "hashtag", "tag": "ai"}, "facebook_hashtag"),
        (
            InstagramTool,
            {"action": "profile_posts", "username": "instagram"},
            "instagram_profile_posts",
        ),
        (
            InstagramTool,
            {"action": "comments", "shortcode": "DeuOR-CjKE9"},
            "instagram_comments",
        ),
        (InstagramTool, {"action": "hashtag", "tag": "ai"}, "instagram_hashtag"),
    ],
)
def test_social_cursor_passthrough(fake_desearch, tool_cls, payload, method):
    _FakeDesearch.next_cursor = NEXT_CURSOR
    first = tool_cls().invoke(payload)

    assert first["next_cursor"] == NEXT_CURSOR
    assert tool_cls().name in first["next_step"]
    called, kwargs = _last_call(fake_desearch)
    assert called == method
    assert "cursor" not in kwargs

    _FakeDesearch.next_cursor = None
    second = tool_cls().invoke({**payload, "cursor": first["next_cursor"]})

    called, kwargs = _last_call(fake_desearch)
    assert called == method
    # The cursor is opaque: it goes back exactly as it came.
    assert kwargs["cursor"] == NEXT_CURSOR
    assert second["next_cursor"] is None
    assert "next_step" not in second


def test_social_cursor_is_not_cleaned(fake_desearch):
    TikTokTool().invoke({"action": "hashtag", "tag": "ai", "cursor": f" {CURSOR} "})
    assert _last_call(fake_desearch)[1]["cursor"] == f" {CURSOR} "


def test_social_empty_strings_count_as_unset(fake_desearch):
    # Some models fill every field; empty ones must not trip the action check.
    TikTokTool().invoke(
        {
            "action": "hashtag",
            "tag": "ai",
            "query": "",
            "username": " ",
            "region": "",
            "cursor": "",
        }
    )
    assert _last_call(fake_desearch)[1] == {"tag": "ai", "include_metadata": True}


def test_social_targets_drop_at_and_hash(fake_desearch):
    TikTokTool().invoke({"action": "profile", "username": " @tiktok "})
    assert _last_call(fake_desearch)[1] == {"username": "tiktok"}
    InstagramTool().invoke({"action": "hashtag", "tag": "#ai"})
    assert _last_call(fake_desearch)[1] == {"tag": "ai", "include_metadata": True}
    FacebookTool().invoke({"action": "profile", "profile_id": "@facebook"})
    assert _last_call(fake_desearch)[1] == {"profile_id": "facebook"}


@pytest.mark.parametrize(
    ("tool_cls", "payload", "message"),
    [
        (TikTokTool, {"action": "comments"}, "tiktok action 'comments' needs video_id"),
        (
            TikTokTool,
            {"action": "comments", "id": "1"},
            "id does not apply to tiktok action 'comments'",
        ),
        (TikTokTool, {"action": "profile", "username": "@"}, "needs username"),
        (TikTokTool, {"action": "search", "query": "   "}, "needs query"),
        (
            TikTokTool,
            {"action": "profile", "username": "tiktok", "count": 3},
            "count does not apply",
        ),
        (
            TikTokTool,
            {"action": "post", "id": "1", "cursor": "abc"},
            "Fields for 'post': id \\(required\\)",
        ),
        (
            TikTokTool,
            {"action": "trending", "cursor": "abc"},
            "'trending' returns one page and does not paginate",
        ),
        (TikTokTool, {"action": "trending", "region": "USA"}, "two-letter code"),
        (
            TikTokTool,
            {"action": "hashtag", "tag": "ai", "count": 0},
            "greater than or equal to 1",
        ),
        (
            TikTokTool,
            {"action": "hashtag", "tag": "ai", "count": 101},
            "less than or equal to 100",
        ),
        (TikTokTool, {"action": "dance"}, "Input should be 'search'"),
        (TikTokTool, {"query": "ai"}, "action"),
        (
            FacebookTool,
            {"action": "profile"},
            "facebook action 'profile' needs profile_id",
        ),
        (FacebookTool, {"action": "trending"}, "Input should be 'search'"),
        (
            FacebookTool,
            {"action": "profile", "username": "facebook"},
            "needs profile_id",
        ),
        (
            InstagramTool,
            {"action": "search", "query": "ai", "cursor": "abc"},
            "'search' returns one page and does not paginate",
        ),
        (
            InstagramTool,
            {"action": "comments", "id": "123"},
            "id does not apply to instagram action 'comments'",
        ),
        (InstagramTool, {"action": "post"}, "instagram action 'post' needs id"),
    ],
)
def test_social_inputs_are_validated_before_any_call(
    fake_desearch, tool_cls, payload, message
):
    with pytest.raises(ValidationError, match=message):
        tool_cls().invoke(payload)
    assert fake_desearch.created == []


def test_social_run_validates_without_the_schema(fake_desearch):
    # Direct _run calls skip the args schema; the action check still applies.
    with pytest.raises(ValueError, match="needs video_id"):
        TikTokTool()._run(action="comments")
    with pytest.raises(ValueError, match="Unknown instagram action 'trending'"):
        InstagramTool()._run(action="trending")
    with pytest.raises(ValueError, match="needs video_id"):
        asyncio.run(TikTokTool()._arun(action="comments"))
    assert fake_desearch.created == []


def test_social_request_unknown_platform():
    with pytest.raises(ValueError, match="Unknown platform 'youtube'"):
        social_request("youtube", "search", query="ai")


def test_social_sdk_errors_are_returned_as_messages(fake_desearch):
    _FakeDesearch.error = RuntimeError("HTTP 422: cursor expired")

    page = TikTokTool().invoke({"action": "hashtag", "tag": "ai", "cursor": CURSOR})
    profile = asyncio.run(
        InstagramTool().ainvoke({"action": "profile", "username": "instagram"})
    )

    assert page == "An error occurred while calling Desearch: HTTP 422: cursor expired"
    assert profile == page
    assert all(client.closed for client in fake_desearch.created)


def test_social_tool_works_inside_a_running_loop(fake_desearch):
    _FakeDesearch.next_cursor = NEXT_CURSOR

    async def _call():
        return FacebookTool().invoke({"action": "hashtag", "tag": "ai"})

    result = asyncio.run(_call())
    assert result["next_cursor"] == NEXT_CURSOR
    assert _last_call(fake_desearch)[0] == "facebook_hashtag"


def test_missing_api_key_raises_after_validation(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        TikTokTool().invoke({"action": "profile", "username": "tiktok"})
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        SearchApiTool().invoke({"query": "Desearch"})
