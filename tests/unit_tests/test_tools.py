import asyncio
import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from langchain_desearch.search_tools import search_tools
from langchain_desearch.tools import (
    BasicTwitterSearchTool,
    BasicWebSearchTool,
    DesearchTool,
    DesearchToolInput,
    FetchLatestTweetsTool,
    FetchRepliesByPostTool,
    FetchRetweetsByPostTool,
    FetchTweetsAndRepliesByUserTool,
    FetchTweetsByIdTool,
    FetchTweetsByUrlsTool,
    FetchTwitterUserTool,
)

ROOT = Path(__file__).resolve().parents[2]
DESEARCH_REQUIREMENT = "desearch-py>=1.1.0,<2.0.0"


class _FakeDesearch:
    created = []

    def __init__(self, api_key: str, base_url: str = "https://api.desearch.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []
        self.closed = False
        self.web_payload = {
            "data": [
                {"title": "one", "link": "https://one.example", "snippet": "1"},
                {"title": "two", "link": "https://two.example", "snippet": "2"},
                {"title": "three", "link": "https://three.example", "snippet": "3"},
            ]
        }
        self.user_posts_payload = {
            "user": {"username": "opentensor"},
            "tweets": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
        }
        self.error = None
        _FakeDesearch.created.append(self)

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return {
            "ai_search": {"text": "answer"},
            "web_search": self.web_payload,
            "x_search": [{"id": "1", "text": "tweet"}],
            "x_posts_by_urls": [{"id": "1"}],
            "x_post_by_id": {"id": "1"},
            "x_user_posts": self.user_posts_payload,
            "x_user_replies": [{"id": "reply"}],
            "x_post_replies": [{"id": "post-reply"}],
            "x_post_retweeters": {"users": [{"id": "u1"}], "next_cursor": None},
        }[name]

    async def ai_search(self, **kwargs):
        return self._record("ai_search", kwargs)

    async def web_search(self, **kwargs):
        return self._record("web_search", kwargs)

    async def x_search(self, **kwargs):
        return self._record("x_search", kwargs)

    async def x_posts_by_urls(self, **kwargs):
        return self._record("x_posts_by_urls", kwargs)

    async def x_post_by_id(self, **kwargs):
        return self._record("x_post_by_id", kwargs)

    async def x_user_posts(self, **kwargs):
        return self._record("x_user_posts", kwargs)

    async def x_user_replies(self, **kwargs):
        return self._record("x_user_replies", kwargs)

    async def x_post_replies(self, **kwargs):
        return self._record("x_post_replies", kwargs)

    async def x_post_retweeters(self, **kwargs):
        return self._record("x_post_retweeters", kwargs)

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_desearch(monkeypatch):
    _FakeDesearch.created = []
    monkeypatch.setenv("DESEARCH_API_KEY", "test-key")
    monkeypatch.setattr("langchain_desearch.tools.Desearch", _FakeDesearch)
    return _FakeDesearch


def test_public_names_stay_compatible():
    assert DesearchTool().name == "desearch_tool"
    assert BasicWebSearchTool().name == "basic_web_search_tool"
    assert BasicTwitterSearchTool().name == "basic_twitter_search_tool"
    assert [tool.name for tool in search_tools] == [
        "desearch_tool",
        "basic_web_search_tool",
        "basic_twitter_search_tool",
    ]


def test_package_metadata_pins_compatible_desearch_py():
    pyproject = (ROOT / "pyproject.toml").read_text()
    setup_py = (ROOT / "setup.py").read_text()
    assert 'version = "1.0.7"' in pyproject
    assert 'version="1.0.7"' in setup_py
    assert DESEARCH_REQUIREMENT in pyproject
    assert DESEARCH_REQUIREMENT in setup_py
    assert "desearch-py>=" in pyproject
    assert "desearch-py>=" in setup_py
    assert "<2.0.0" in pyproject
    assert "<2.0.0" in setup_py


def _bind(method, **kwargs):
    """Fail if a tool would send arguments the installed SDK does not accept."""
    inspect.signature(method).bind(object(), **kwargs)


def test_installed_sdk_accepts_tool_calls():
    from desearch_py import Desearch

    assert hasattr(Desearch, "web_search")
    assert hasattr(Desearch, "x_search")
    assert hasattr(Desearch, "ai_search")
    assert not hasattr(Desearch, "basic_web_search")
    assert not hasattr(Desearch, "basic_twitter_search")

    _bind(Desearch.web_search, query="Bittensor", start=0)
    _bind(
        Desearch.ai_search,
        prompt="What is Bittensor?",
        tools=["web"],
        date_filter=None,
    )
    _bind(Desearch.x_search, query="Bittensor", sort="Top", count=10)
    _bind(Desearch.x_posts_by_urls, urls=["https://x.com/opentensor/status/1"])
    _bind(Desearch.x_post_by_id, id="1")
    _bind(Desearch.x_user_posts, username="opentensor")
    _bind(Desearch.x_user_replies, user="opentensor", query="tao", count=10)
    _bind(Desearch.x_post_replies, post_id="1", query=None, count=10)
    _bind(Desearch.x_post_retweeters, id="1", cursor=None)


def test_empty_tool_list_is_rejected_by_pydantic_v2_validator():
    with pytest.raises(ValidationError, match="at least one valid tool"):
        DesearchToolInput(prompt="What is Bittensor?", tool=[])

    with pytest.raises(ValidationError, match="at least one valid tool"):
        DesearchTool().invoke({"prompt": "What is Bittensor?", "tool": []})


def test_desearch_tool_invoke_calls_ai_search(fake_desearch):
    result = DesearchTool().invoke(
        {"prompt": "What is Bittensor?", "tool": ["web"], "model": "NOVA"}
    )

    assert result == {"text": "answer"}
    client = fake_desearch.created[-1]
    assert client.closed is True
    assert client.api_key == "test-key"
    method, kwargs = client.calls[-1]
    assert method == "ai_search"
    assert kwargs["prompt"] == "What is Bittensor?"
    assert kwargs["tools"] == ["web"]
    assert kwargs["date_filter"] is None
    assert "model" not in kwargs
    assert "streaming" not in kwargs


def test_desearch_tool_rejects_unknown_model(fake_desearch):
    with pytest.raises(ValueError, match="Model should be"):
        DesearchTool().invoke(
            {"prompt": "What is Bittensor?", "tool": ["web"], "model": "GPT"}
        )
    assert fake_desearch.created == []


def test_desearch_tool_forwards_current_ai_search_params(fake_desearch):
    DesearchTool().invoke(
        {
            "prompt": "Bittensor",
            "tool": ["web", "reddit"],
            "model": "ORBIT",
            "date_filter": "PAST_WEEK",
            "count": 10,
            "result_type": "ONLY_LINKS",
            "streaming": True,
        }
    )
    _, kwargs = fake_desearch.created[-1].calls[-1]
    assert kwargs["date_filter"] == "PAST_WEEK"
    assert kwargs["count"] == 10
    assert kwargs["result_type"] == "ONLY_LINKS"
    assert "streaming" not in kwargs
    assert "model" not in kwargs


def test_basic_web_search_uses_web_search_and_trims_results(fake_desearch):
    result = BasicWebSearchTool().invoke(
        {"query": "Bittensor subnet 22 Desearch", "num": 2}
    )

    assert result["data"] == [
        {"title": "one", "link": "https://one.example", "snippet": "1"},
        {"title": "two", "link": "https://two.example", "snippet": "2"},
    ]
    method, kwargs = fake_desearch.created[-1].calls[-1]
    assert method == "web_search"
    assert kwargs == {"query": "Bittensor subnet 22 Desearch", "start": 0}
    assert fake_desearch.created[-1].closed is True


def test_web_search_start_is_one_based(fake_desearch):
    BasicWebSearchTool().invoke({"query": "ai", "num": 10, "start": 1})
    assert fake_desearch.created[-1].calls[-1][1]["start"] == 0

    BasicWebSearchTool().invoke({"query": "ai", "num": 10, "start": 11})
    assert fake_desearch.created[-1].calls[-1][1]["start"] == 10


def test_web_search_serializes_pydantic_models(fake_desearch):
    class _Page(BaseModel):
        data: list

    client_holder = {}

    class _ModelClient(_FakeDesearch):
        async def web_search(self, **kwargs):
            client_holder["client"] = self
            self.calls.append(("web_search", kwargs))
            return _Page(
                data=[{"title": "A", "link": "https://a.example", "snippet": "s"}]
            )

    fake_desearch.created = []
    # The fixture patches Desearch; replace it with the model-returning client.
    import langchain_desearch.tools as tools

    tools.Desearch = _ModelClient
    try:
        result = BasicWebSearchTool().invoke({"query": "ai", "num": 5, "start": 1})
    finally:
        tools.Desearch = _FakeDesearch

    assert result == {
        "data": [{"title": "A", "link": "https://a.example", "snippet": "s"}]
    }
    assert client_holder["client"].closed is True


def test_basic_twitter_search_uses_x_search(fake_desearch):
    result = BasicTwitterSearchTool().invoke(
        {
            "query": "Bittensor",
            "count": 10,
            "sort": "Top",
            "user": "opentensor",
            "lang": "en",
            "verified": True,
            "min_likes": 1,
        }
    )

    assert result == [{"id": "1", "text": "tweet"}]
    method, kwargs = fake_desearch.created[-1].calls[-1]
    assert method == "x_search"
    assert kwargs["query"] == "Bittensor"
    assert kwargs["count"] == 10
    assert kwargs["sort"] == "Top"
    assert kwargs["user"] == "opentensor"
    assert kwargs["lang"] == "en"
    assert kwargs["verified"] is True
    assert kwargs["min_likes"] == 1
    assert "start_date" not in kwargs


@pytest.mark.parametrize(
    ("tool_cls", "payload", "method", "expected"),
    [
        (
            FetchTweetsByUrlsTool,
            {"urls": ["https://x.com/opentensor/status/1"]},
            "x_posts_by_urls",
            {"urls": ["https://x.com/opentensor/status/1"]},
        ),
        (FetchTweetsByIdTool, {"id": "99"}, "x_post_by_id", {"id": "99"}),
        (
            FetchTweetsAndRepliesByUserTool,
            {"user": "opentensor", "query": "tao", "count": 4},
            "x_user_replies",
            {"user": "opentensor", "query": "tao", "count": 4},
        ),
        (
            FetchRepliesByPostTool,
            {"post_id": "99", "count": 3},
            "x_post_replies",
            {"post_id": "99", "count": 3},
        ),
        (
            FetchRetweetsByPostTool,
            {"post_id": "99", "query": "ignored", "count": 2, "cursor": "abc"},
            "x_post_retweeters",
            {"id": "99", "cursor": "abc"},
        ),
        (FetchTwitterUserTool, {"user": "opentensor"}, "x_user_posts", {"username": "opentensor"}),
    ],
)
def test_remaining_tools_use_current_methods(
    fake_desearch, tool_cls, payload, method, expected
):
    tool_cls().invoke(payload)
    called, kwargs = fake_desearch.created[-1].calls[-1]
    assert called == method
    assert kwargs == expected
    assert fake_desearch.created[-1].closed is True


def test_latest_tweets_trims_user_timeline(fake_desearch):
    result = FetchLatestTweetsTool().invoke({"user": "opentensor", "count": 2})
    assert result["tweets"] == [{"id": "1"}, {"id": "2"}]
    assert fake_desearch.created[-1].calls[-1] == ("x_user_posts", {"username": "opentensor"})


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DESEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DESEARCH_API_KEY"):
        BasicWebSearchTool().invoke({"query": "ai"})


def test_sdk_errors_are_returned_as_messages(fake_desearch):
    fake_desearch.created = []

    class _Boom(_FakeDesearch):
        async def web_search(self, **kwargs):
            self.calls.append(("web_search", kwargs))
            raise RuntimeError("boom")

    import langchain_desearch.tools as tools

    tools.Desearch = _Boom
    try:
        result = BasicWebSearchTool().invoke({"query": "ai"})
    finally:
        tools.Desearch = _FakeDesearch

    assert result == "An error occurred while calling Desearch: boom"
    assert _Boom.created[-1].closed is True


def test_sync_invoke_works_inside_a_running_loop(fake_desearch):
    async def _call():
        return BasicTwitterSearchTool().invoke({"query": "Bittensor", "count": 10})

    result = asyncio.run(_call())
    assert result == [{"id": "1", "text": "tweet"}]
    assert fake_desearch.created[-1].calls[-1][0] == "x_search"


def test_async_invoke_awaits_the_sdk(fake_desearch):
    result = asyncio.run(
        DesearchTool().ainvoke({"prompt": "What is Bittensor?", "tool": ["web"]})
    )
    assert result == {"text": "answer"}
    assert fake_desearch.created[-1].calls[-1][0] == "ai_search"
    assert fake_desearch.created[-1].closed is True
