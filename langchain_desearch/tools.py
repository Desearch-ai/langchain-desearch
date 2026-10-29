import asyncio
import concurrent.futures
import os
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from desearch_py import Desearch
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema

from langchain_desearch.people import (
    PEOPLE_API_BASE_URL,
    PeopleClient,
    list_people_profiles,
    people_search_body,
    search_people,
)

_AI_SEARCH_MODELS = ("NOVA", "ORBIT", "HORIZON")


def _require_api_key() -> str:
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise ValueError("DESEARCH_API_KEY environment variable not set.")
    return api_key


def _compact(**kwargs: Any) -> Dict[str, Any]:
    """Drop unset optional arguments so SDK defaults stay intact."""
    return {key: value for key, value in kwargs.items() if value is not None}


def _to_plain(value: Any) -> Any:
    """Convert desearch-py pydantic models into JSON-friendly values."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _run_coroutine(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Run an async SDK call from sync tool code, including inside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


async def _invoke_client(api_key: str, method_name: str, **kwargs: Any) -> Any:
    client = Desearch(api_key=api_key)
    try:
        method = getattr(client, method_name)
        result = await method(**kwargs)
    finally:
        await client.close()
    return _to_plain(result)


def _call_client(method_name: str, **kwargs: Any) -> Any:
    api_key = _require_api_key()
    try:
        return _run_coroutine(lambda: _invoke_client(api_key, method_name, **kwargs))
    except Exception as exc:
        return f"An error occurred while calling Desearch: {exc}"


async def _acall_client(method_name: str, **kwargs: Any) -> Any:
    api_key = _require_api_key()
    try:
        return await _invoke_client(api_key, method_name, **kwargs)
    except Exception as exc:
        return f"An error occurred while calling Desearch: {exc}"


def _limit_web_results(result: Any, num: int) -> Any:
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        return {**result, "data": result["data"][:num]}
    return result


def _limit_user_posts(result: Any, count: int) -> Any:
    if isinstance(result, dict) and isinstance(result.get("tweets"), list):
        return {**result, "tweets": result["tweets"][:count]}
    return result


def _web_search_offset(start: int) -> int:
    """Map the tool's 1-based start index onto web_search's skip offset."""
    return max(int(start) - 1, 0)


def _validate_ai_model(model: str) -> None:
    if model not in _AI_SEARCH_MODELS:
        raise ValueError("Model should be 'NOVA', 'ORBIT' or 'HORIZON'")


def _ai_search_kwargs(
    prompt: str,
    tool: List[str],
    date_filter: Optional[str],
    count: Optional[int],
    result_type: Optional[str],
    system_message: Optional[str],
    scoring_system_message: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Any]:
    # Pass date_filter even when it is None. desearch-py otherwise defaults
    # the argument to PAST_24_HOURS; an explicit None omits the filter.
    kwargs = _compact(
        prompt=prompt,
        tools=tool,
        count=count,
        result_type=result_type,
        system_message=system_message,
        scoring_system_message=scoring_system_message,
        start_date=start_date,
        end_date=end_date,
    )
    kwargs["date_filter"] = date_filter
    return kwargs


class DesearchToolInput(BaseModel):
    prompt: str = Field(description="The search prompt or query.")
    tool: List[
        Literal[
            "web", "hackernews", "reddit", "wikipedia", "youtube", "twitter", "arxiv"
        ]
    ] = Field(description="List of tools to use. Must include at least one tool.")
    model: str = Field(
        default="NOVA",
        description=(
            "Accepted for compatibility. Must be 'NOVA', 'ORBIT', or 'HORIZON'. "
            "The current Desearch SDK does not send a model parameter."
        ),
    )
    date_filter: Optional[str] = Field(
        default=None, description="Date filter for the search."
    )
    streaming: Optional[bool] = Field(
        default=False,
        description=(
            "Accepted for compatibility. desearch-py always requests a "
            "non-streaming AI search response."
        ),
    )
    count: Optional[int] = Field(
        default=None, description="Number of results to return per source (10-200)."
    )
    result_type: Optional[str] = Field(
        default=None,
        description="Result type: ONLY_LINKS or LINKS_WITH_FINAL_SUMMARY.",
    )
    system_message: Optional[str] = Field(
        default=None, description="System message for the search."
    )
    scoring_system_message: Optional[str] = Field(
        default=None, description="System message used when scoring the response."
    )
    start_date: Optional[str] = Field(
        default=None, description="Start date in UTC (YYYY-MM-DDTHH:MM:SSZ)."
    )
    end_date: Optional[str] = Field(
        default=None, description="End date in UTC (YYYY-MM-DDTHH:MM:SSZ)."
    )

    @model_validator(mode="after")
    def check_tool_non_empty(self) -> "DesearchToolInput":
        if not self.tool:
            raise ValueError("The 'tool' field must contain at least one valid tool.")
        return self


class DesearchTool(BaseTool):
    name: str = "desearch_tool"
    description: str = (
        "Performs different Desearch API searches like AI search, Web links search, and Twitter posts search."
    )
    args_schema: ArgsSchema = DesearchToolInput
    return_direct: bool = True

    def _run(
        self,
        prompt: str,
        tool: List[str],
        model: str = "NOVA",
        date_filter: Optional[str] = None,
        streaming: bool = False,
        count: Optional[int] = None,
        result_type: Optional[str] = None,
        system_message: Optional[str] = None,
        scoring_system_message: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del streaming, run_manager
        _validate_ai_model(model)
        return _call_client(
            "ai_search",
            **_ai_search_kwargs(
                prompt,
                tool,
                date_filter,
                count,
                result_type,
                system_message,
                scoring_system_message,
                start_date,
                end_date,
            ),
        )

    async def _arun(
        self,
        prompt: str,
        tool: List[str],
        model: str = "NOVA",
        date_filter: Optional[str] = None,
        streaming: bool = False,
        count: Optional[int] = None,
        result_type: Optional[str] = None,
        system_message: Optional[str] = None,
        scoring_system_message: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del streaming, run_manager
        _validate_ai_model(model)
        return await _acall_client(
            "ai_search",
            **_ai_search_kwargs(
                prompt,
                tool,
                date_filter,
                count,
                result_type,
                system_message,
                scoring_system_message,
                start_date,
                end_date,
            ),
        )


class BasicWebSearchToolInput(BaseModel):
    query: str = Field(description="The search query.")
    num: int = Field(
        default=10,
        description=(
            "Maximum number of results to return. desearch-py web_search has no "
            "result-count parameter, so this trims the returned page."
        ),
    )
    start: int = Field(
        default=1,
        description=(
            "1-based index of the first result. Passed to desearch-py web_search "
            "as a zero-based skip offset."
        ),
    )


class BasicWebSearchTool(BaseTool):
    name: str = "basic_web_search_tool"
    description: str = "Performs a basic web search using Desearch."
    args_schema: ArgsSchema = BasicWebSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        result = _call_client(
            "web_search", query=query, start=_web_search_offset(start)
        )
        return _limit_web_results(result, num)

    async def _arun(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        result = await _acall_client(
            "web_search", query=query, start=_web_search_offset(start)
        )
        return _limit_web_results(result, num)


class BasicTwitterSearchToolInput(BaseModel):
    query: str = Field(description="The Twitter search query.")
    sort: str = Field(default="Top", description="Sort order: 'Top' or 'Latest'.")
    count: int = Field(default=10, description="Number of results to return.")
    user: Optional[str] = Field(default=None, description="User to search for.")
    start_date: Optional[str] = Field(
        default=None, description="Start date in UTC (YYYY-MM-DD)."
    )
    end_date: Optional[str] = Field(
        default=None, description="End date in UTC (YYYY-MM-DD)."
    )
    lang: Optional[str] = Field(
        default=None, description="Language code (for example en, es, fr)."
    )
    verified: Optional[bool] = Field(
        default=None, description="Filter for verified users."
    )
    blue_verified: Optional[bool] = Field(
        default=None, description="Filter for blue-checkmark verified users."
    )
    is_quote: Optional[bool] = Field(
        default=None, description="Include only tweets with quotes."
    )
    is_video: Optional[bool] = Field(
        default=None, description="Include only tweets with videos."
    )
    is_image: Optional[bool] = Field(
        default=None, description="Include only tweets with images."
    )
    min_retweets: Optional[int] = Field(
        default=None, description="Minimum number of retweets."
    )
    min_replies: Optional[int] = Field(
        default=None, description="Minimum number of replies."
    )
    min_likes: Optional[int] = Field(
        default=None, description="Minimum number of likes."
    )


class BasicTwitterSearchTool(BaseTool):
    name: str = "basic_twitter_search_tool"
    description: str = "Performs a basic Twitter search using Desearch."
    args_schema: ArgsSchema = BasicTwitterSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        query: str,
        sort: str = "Top",
        count: int = 10,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lang: Optional[str] = None,
        verified: Optional[bool] = None,
        blue_verified: Optional[bool] = None,
        is_quote: Optional[bool] = None,
        is_video: Optional[bool] = None,
        is_image: Optional[bool] = None,
        min_retweets: Optional[int] = None,
        min_replies: Optional[int] = None,
        min_likes: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "x_search",
            **_compact(
                query=query,
                sort=sort,
                count=count,
                user=user,
                start_date=start_date,
                end_date=end_date,
                lang=lang,
                verified=verified,
                blue_verified=blue_verified,
                is_quote=is_quote,
                is_video=is_video,
                is_image=is_image,
                min_retweets=min_retweets,
                min_replies=min_replies,
                min_likes=min_likes,
            ),
        )

    async def _arun(
        self,
        query: str,
        sort: str = "Top",
        count: int = 10,
        user: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lang: Optional[str] = None,
        verified: Optional[bool] = None,
        blue_verified: Optional[bool] = None,
        is_quote: Optional[bool] = None,
        is_video: Optional[bool] = None,
        is_image: Optional[bool] = None,
        min_retweets: Optional[int] = None,
        min_replies: Optional[int] = None,
        min_likes: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "x_search",
            **_compact(
                query=query,
                sort=sort,
                count=count,
                user=user,
                start_date=start_date,
                end_date=end_date,
                lang=lang,
                verified=verified,
                blue_verified=blue_verified,
                is_quote=is_quote,
                is_video=is_video,
                is_image=is_image,
                min_retweets=min_retweets,
                min_replies=min_replies,
                min_likes=min_likes,
            ),
        )


class FetchTweetsByUrlsToolInput(BaseModel):
    urls: list = Field(description="A list of Twitter URLs to fetch tweets from.")


class FetchTweetsByUrlsTool(BaseTool):
    name: str = "fetch_tweets_by_urls_tool"
    description: str = "Fetch tweets from the provided URLs."
    args_schema: ArgsSchema = FetchTweetsByUrlsToolInput
    return_direct: bool = True

    def _run(
        self,
        urls: list,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client("x_posts_by_urls", urls=urls)

    async def _arun(
        self,
        urls: list,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client("x_posts_by_urls", urls=urls)


class FetchTweetsByIdToolInput(BaseModel):
    id: str = Field(description="The ID of the tweet to fetch.")


class FetchTweetsByIdTool(BaseTool):
    name: str = "fetch_tweets_by_id_tool"
    description: str = "Fetch a tweet by its ID."
    args_schema: ArgsSchema = FetchTweetsByIdToolInput
    return_direct: bool = True

    def _run(
        self,
        id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client("x_post_by_id", id=id)

    async def _arun(
        self,
        id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client("x_post_by_id", id=id)


class FetchLatestTweetsToolInput(BaseModel):
    user: str = Field(description="The username to fetch the latest tweets from.")
    count: int = Field(default=10, description="Number of tweets to fetch.")


class FetchLatestTweetsTool(BaseTool):
    name: str = "fetch_latest_tweets_tool"
    description: str = "Fetch the latest tweets from a user."
    args_schema: ArgsSchema = FetchLatestTweetsToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _limit_user_posts(_call_client("x_user_posts", username=user), count)

    async def _arun(
        self,
        user: str,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return _limit_user_posts(
            await _acall_client("x_user_posts", username=user), count
        )


class FetchTweetsAndRepliesByUserToolInput(BaseModel):
    user: str = Field(description="The username to fetch tweets and replies from.")
    query: Optional[str] = Field(
        default=None, description="Query to filter tweets and replies."
    )
    count: int = Field(default=10, description="Number of tweets and replies to fetch.")


class FetchTweetsAndRepliesByUserTool(BaseTool):
    name: str = "fetch_tweets_and_replies_by_user_tool"
    description: str = "Fetch tweets and replies by a specific user."
    args_schema: ArgsSchema = FetchTweetsAndRepliesByUserToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "x_user_replies", **_compact(user=user, query=query, count=count)
        )

    async def _arun(
        self,
        user: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "x_user_replies", **_compact(user=user, query=query, count=count)
        )


class FetchRepliesByPostToolInput(BaseModel):
    post_id: str = Field(description="The ID of the post to fetch replies for.")
    query: Optional[str] = Field(default=None, description="Query to filter replies.")
    count: int = Field(default=10, description="Number of replies to fetch.")


class FetchRepliesByPostTool(BaseTool):
    name: str = "fetch_replies_by_post_tool"
    description: str = "Fetch replies to a specific post."
    args_schema: ArgsSchema = FetchRepliesByPostToolInput
    return_direct: bool = True

    def _run(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "x_post_replies", **_compact(post_id=post_id, query=query, count=count)
        )

    async def _arun(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "x_post_replies", **_compact(post_id=post_id, query=query, count=count)
        )


class FetchRetweetsByPostToolInput(BaseModel):
    post_id: str = Field(description="The ID of the post to fetch retweets for.")
    query: Optional[str] = Field(default=None, description="Query to filter retweets.")
    count: int = Field(default=10, description="Number of retweets to fetch.")
    cursor: Optional[str] = Field(
        default=None, description="Pagination cursor for the retweeter list."
    )


class FetchRetweetsByPostTool(BaseTool):
    name: str = "fetch_retweets_by_post_tool"
    description: str = "Fetch users who retweeted a specific post."
    args_schema: ArgsSchema = FetchRetweetsByPostToolInput
    return_direct: bool = True

    def _run(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        # query and count stay in the signature for older callers. The current
        # SDK pages retweeters by id and cursor.
        del query, count, run_manager
        return _call_client("x_post_retweeters", **_compact(id=post_id, cursor=cursor))

    async def _arun(
        self,
        post_id: str,
        query: Optional[str] = None,
        count: int = 10,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del query, count, run_manager
        return await _acall_client(
            "x_post_retweeters", **_compact(id=post_id, cursor=cursor)
        )


class FetchTwitterUserToolInput(BaseModel):
    user: str = Field(description="The username to fetch information about.")


class FetchTwitterUserTool(BaseTool):
    name: str = "fetch_twitter_user_tool"
    description: str = "Fetch information about a specific Twitter user."
    args_schema: ArgsSchema = FetchTwitterUserToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client("x_user_posts", username=user)

    async def _arun(
        self,
        user: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client("x_user_posts", username=user)


# ─── People Search ───────────────────────────────────────────────────────────
#
# desearch-py ships People Search in 1.4.0, which is not on PyPI yet, so these
# tools call the /desearch/people routes over HTTP (see people.py).

_PeoplePlatform = Literal["x", "youtube", "linkedin", "tiktok"]
_PeopleChannel = Literal[
    "email", "website", "x", "linkedin", "instagram", "youtube", "tiktok"
]
_PEOPLE_FILTER_FIELDS = (
    "min_followers",
    "max_followers",
    "min_subscribers",
    "max_subscribers",
    "verified_only",
    "require_contact",
    "countries",
    "keywords",
    "exclude_keywords",
)


def _people_error(exc: Exception) -> str:
    detail = str(exc) or type(exc).__name__
    return f"An error occurred while calling Desearch: {detail}"


async def _invoke_people(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    async with PeopleClient(api_key, base_url) as client:
        return await operation(client, **kwargs)


def _call_people(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return _run_coroutine(
            lambda: _invoke_people(api_key, base_url, operation, **kwargs)
        )
    except Exception as exc:
        return _people_error(exc)


async def _acall_people(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return await _invoke_people(api_key, base_url, operation, **kwargs)
    except Exception as exc:
        return _people_error(exc)


class PeopleSearchToolInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=1000,
        description=(
            "Who to find, in plain language, for example 'Heads of sales at fintech "
            "startups who post about outbound'. Required to start a search."
        ),
    )
    platform: Optional[_PeoplePlatform] = Field(
        default=None,
        description=(
            "Where to search: 'x' (X accounts), 'linkedin' (LinkedIn members), "
            "'youtube' (YouTube creators) or 'tiktok' (TikTok creators). "
            "Required to start a search."
        ),
    )
    count: int = Field(
        default=25,
        ge=5,
        le=200,
        description=(
            "How many qualified people to find (5-200). The search reserves this "
            "many profiles at the route price and bills only the ones it delivers."
        ),
    )
    min_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Only return people with at least this fit score (0-100).",
    )
    exclude: Optional[List[str]] = Field(
        default=None,
        description="Handles to leave out, for example people you already contacted.",
    )
    min_followers: Optional[int] = Field(
        default=None, ge=0, description="Minimum followers (x, linkedin, tiktok)."
    )
    max_followers: Optional[int] = Field(
        default=None, ge=0, description="Maximum followers (x, linkedin, tiktok)."
    )
    min_subscribers: Optional[int] = Field(
        default=None, ge=0, description="Minimum subscribers (youtube)."
    )
    max_subscribers: Optional[int] = Field(
        default=None, ge=0, description="Maximum subscribers (youtube)."
    )
    verified_only: Optional[bool] = Field(
        default=None, description="Only verified or X Premium accounts (x)."
    )
    require_contact: Optional[bool] = Field(
        default=None,
        description="Only creators with a public email or contact link (youtube, tiktok).",
    )
    countries: Optional[List[str]] = Field(
        default=None,
        description="Countries to prefer, for example ['US', 'GB'] (x, youtube, linkedin).",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Words that should appear in their posts, videos or bios.",
    )
    exclude_keywords: Optional[List[str]] = Field(
        default=None, description="Leave out people who mention these words."
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Any unique string. Sending the same key again returns the first "
            "search instead of starting and billing a new one."
        ),
    )
    timeout: int = Field(
        default=300,
        ge=0,
        le=1800,
        description=(
            "Seconds to wait for the search to finish (default 300). If it is "
            "still running, the people found so far are returned with the task_id."
        ),
    )
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "task_id of a search started earlier. Waits on that search and returns "
            "its people instead of starting a new one. Free."
        ),
    )

    @model_validator(mode="after")
    def check_search_or_task(self) -> "PeopleSearchToolInput":
        if self.task_id:
            return self
        if not self.query or not self.platform:
            raise ValueError(
                "Set query and platform to start a People Search, or task_id to "
                "keep waiting on one."
            )
        # Refuse filters the platform does not accept before anything is billed.
        people_search_body(
            self.platform,
            self.query,
            **{name: getattr(self, name) for name in _PEOPLE_FILTER_FIELDS},
        )
        return self


class PeopleSearchTool(BaseTool):
    name: str = "people_search_tool"
    description: str = (
        "Finds people on X, LinkedIn, YouTube or TikTok who match a plain-language "
        "description (for example 'founders of AI developer-tool startups who post "
        "about hiring') and returns compact profiles: name, handle, url, platform, "
        "audience, location, fit score, why they matched, contacts and links. "
        "BILLED: starting a search reserves `count` profiles at the Desearch route "
        "price; you pay only for the profiles it delivers, and a search that finds "
        "no one is free. SLOW: a search runs for 1-5 minutes. This tool waits up to "
        "`timeout` seconds, then returns the people found so far and the task_id. "
        "To keep waiting, call it again with only task_id (free; it does not start "
        "or bill a new search). Check people_profiles_tool first for people earlier "
        "searches already found."
    )
    args_schema: ArgsSchema = PeopleSearchToolInput
    return_direct: bool = True
    base_url: str = PEOPLE_API_BASE_URL
    poll_interval: float = 5.0

    def _search_kwargs(
        self,
        query: Optional[str],
        platform: Optional[str],
        count: int,
        min_score: Optional[int],
        exclude: Optional[List[str]],
        filters: Dict[str, Any],
        idempotency_key: Optional[str],
        timeout: int,
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "tool_name": self.name,
            "timeout": timeout,
            "poll_interval": self.poll_interval,
        }
        if task_id:
            return {**kwargs, "task_id": task_id}
        if not query or not platform:
            raise ValueError(
                "Set query and platform to start a People Search, or task_id to "
                "keep waiting on one."
            )
        body = people_search_body(
            platform,
            query,
            count=count,
            min_score=min_score,
            exclude=exclude,
            **filters,
        )
        return {
            **kwargs,
            "platform": platform,
            "body": body,
            "idempotency_key": idempotency_key,
        }

    def _run(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        count: int = 25,
        min_score: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_subscribers: Optional[int] = None,
        max_subscribers: Optional[int] = None,
        verified_only: Optional[bool] = None,
        require_contact: Optional[bool] = None,
        countries: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        kwargs = self._search_kwargs(
            query,
            platform,
            count,
            min_score,
            exclude,
            _compact(
                min_followers=min_followers,
                max_followers=max_followers,
                min_subscribers=min_subscribers,
                max_subscribers=max_subscribers,
                verified_only=verified_only,
                require_contact=require_contact,
                countries=countries,
                keywords=keywords,
                exclude_keywords=exclude_keywords,
            ),
            idempotency_key,
            timeout,
            task_id,
        )
        return _call_people(_require_api_key(), self.base_url, search_people, **kwargs)

    async def _arun(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        count: int = 25,
        min_score: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_subscribers: Optional[int] = None,
        max_subscribers: Optional[int] = None,
        verified_only: Optional[bool] = None,
        require_contact: Optional[bool] = None,
        countries: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        kwargs = self._search_kwargs(
            query,
            platform,
            count,
            min_score,
            exclude,
            _compact(
                min_followers=min_followers,
                max_followers=max_followers,
                min_subscribers=min_subscribers,
                max_subscribers=max_subscribers,
                verified_only=verified_only,
                require_contact=require_contact,
                countries=countries,
                keywords=keywords,
                exclude_keywords=exclude_keywords,
            ),
            idempotency_key,
            timeout,
            task_id,
        )
        return await _acall_people(
            _require_api_key(), self.base_url, search_people, **kwargs
        )


class PeopleProfilesToolInput(BaseModel):
    platform: _PeoplePlatform = Field(
        description="Whose people to list: 'x', 'youtube', 'linkedin' or 'tiktok'."
    )
    q: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Text in the name, handle, bio or location.",
    )
    sort: Literal["score", "recent", "audience"] = Field(
        default="score",
        description="'score' (best fit first), 'recent' or 'audience' (largest first).",
    )
    limit: int = Field(
        default=25, ge=1, le=200, description="People per page (1-200)."
    )
    cursor: Optional[str] = Field(
        default=None, description="next_cursor from the previous page."
    )
    has: Optional[List[_PeopleChannel]] = Field(
        default=None,
        description=(
            "Channels every person must have, for example ['email'] or "
            "['email', 'instagram']."
        ),
    )
    min_audience: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum followers (subscribers on YouTube).",
    )
    max_audience: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum followers (subscribers on YouTube).",
    )
    min_score: Optional[int] = Field(
        default=None, ge=0, le=100, description="Minimum fit score (0-100)."
    )
    location: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Text in the location, for example 'united states'.",
    )
    verified: Optional[bool] = Field(
        default=None, description="True keeps only verified accounts (x, tiktok)."
    )


class PeopleProfilesTool(BaseTool):
    name: str = "people_profiles_tool"
    description: str = (
        "Lists the people that earlier Desearch People Searches already found on "
        "one platform (x, youtube, linkedin or tiktok), one entry per person with "
        "their best fit score and the search that found them. Filter by text, by "
        "channels they have (has=email, website, x, linkedin, instagram, youtube, "
        "tiktok), audience range, minimum score, location and verified (x, tiktok); "
        "sort by score, recent or audience. Free and instant: it reads saved "
        "results and never starts a search."
    )
    args_schema: ArgsSchema = PeopleProfilesToolInput
    return_direct: bool = True
    base_url: str = PEOPLE_API_BASE_URL

    def _run(
        self,
        platform: str,
        q: Optional[str] = None,
        sort: str = "score",
        limit: int = 25,
        cursor: Optional[str] = None,
        has: Optional[List[str]] = None,
        min_audience: Optional[int] = None,
        max_audience: Optional[int] = None,
        min_score: Optional[int] = None,
        location: Optional[str] = None,
        verified: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_people(
            _require_api_key(),
            self.base_url,
            list_people_profiles,
            tool_name=self.name,
            platform=platform,
            q=q,
            sort=sort,
            limit=limit,
            cursor=cursor,
            has=has,
            min_audience=min_audience,
            max_audience=max_audience,
            min_score=min_score,
            location=location,
            verified=verified,
        )

    async def _arun(
        self,
        platform: str,
        q: Optional[str] = None,
        sort: str = "score",
        limit: int = 25,
        cursor: Optional[str] = None,
        has: Optional[List[str]] = None,
        min_audience: Optional[int] = None,
        max_audience: Optional[int] = None,
        min_score: Optional[int] = None,
        location: Optional[str] = None,
        verified: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_people(
            _require_api_key(),
            self.base_url,
            list_people_profiles,
            tool_name=self.name,
            platform=platform,
            q=q,
            sort=sort,
            limit=limit,
            cursor=cursor,
            has=has,
            min_audience=min_audience,
            max_audience=max_audience,
            min_score=min_score,
            location=location,
            verified=verified,
        )
