import asyncio
import concurrent.futures
import inspect
import os
import re
from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from desearch_py import Desearch, FacebookPost
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema

from langchain_desearch.companies import (
    COMPANY_API_BASE_URL,
    CompanyClient,
    company_enrich_body,
    company_search_body,
    control_company_task,
    list_company_tasks,
    run_company_task,
)
from langchain_desearch.monitors import (
    MONITOR_API_BASE_URL,
    MONITOR_CONTROL_TOOL,
    MONITOR_CREATE_TOOL,
    MONITOR_ESTIMATE_TOOL,
    MONITOR_ITEMS_TOOL,
    MONITOR_RUNS_TOOL,
    MONITOR_UPDATE_TOOL,
    MONITORS_TOOL,
    MonitorClient,
    control_monitor,
    create_monitor,
    estimate_monitor,
    get_monitors,
    monitor_create_body,
    monitor_estimate_body,
    monitor_options,
    monitor_settings,
    monitor_webhook,
    read_items,
    read_runs,
    update_monitor,
)
from langchain_desearch.people import (
    PEOPLE_API_BASE_URL,
    PeopleClient,
    list_people_profiles,
    people_search_body,
    search_people,
)
from langchain_desearch.social import social_page, social_request

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


# POST /desearch/facebook/search takes `sort` (desearch-public-api#644); desearch-py's
# facebook_search takes it from 1.13.0.
_FACEBOOK_SEARCH_TAKES_SORT = "sort" in inspect.signature(Desearch.facebook_search).parameters


async def _invoke_client(api_key: str, method_name: str, **kwargs: Any) -> Any:
    client = Desearch(api_key=api_key)
    try:
        if method_name == "facebook_search" and "sort" in kwargs and not _FACEBOOK_SEARCH_TAKES_SORT:
            raise ValueError("Facebook search sort needs desearch-py 1.13.0 or later")
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


class SearchTweetsByUserToolInput(BaseModel):
    user: str = Field(
        min_length=1,
        description="X username whose posts to search, for example elonmusk.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Words or X search operators that the user's posts must match.",
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of posts to return (1-100). The API default is 10.",
    )


class SearchTweetsByUserTool(BaseTool):
    name: str = "search_tweets_by_user_tool"
    description: str = (
        "Search the posts of one X (Twitter) user, optionally only those matching "
        "a query. Returns a list of posts."
    )
    args_schema: ArgsSchema = SearchTweetsByUserToolInput
    return_direct: bool = True

    def _run(
        self,
        user: str,
        query: Optional[str] = None,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "x_posts_by_user", **_compact(user=user, query=query, count=count)
        )

    async def _arun(
        self,
        user: str,
        query: Optional[str] = None,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "x_posts_by_user", **_compact(user=user, query=query, count=count)
        )


class FetchTwitterTrendsToolInput(BaseModel):
    woeid: int = Field(
        default=1,
        description=(
            "WOEID of the location: Worldwide is 1 (the default) and the United "
            "States is 23424977. Only locations X publishes trends for are accepted."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=30,
        le=100,
        description="Number of trends to return (30-100). The API default is 30.",
    )


class FetchTwitterTrendsTool(BaseTool):
    name: str = "fetch_twitter_trends_tool"
    description: str = (
        "Fetch the trending topics on X (Twitter) for a location, by WOEID "
        "(Worldwide is 1, the United States is 23424977). Returns the trends "
        "with their rank and search query, and the location."
    )
    args_schema: ArgsSchema = FetchTwitterTrendsToolInput
    return_direct: bool = True

    def _run(
        self,
        woeid: int = 1,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client("x_trends", **_compact(woeid=woeid, count=count))

    async def _arun(
        self,
        woeid: int = 1,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client("x_trends", **_compact(woeid=woeid, count=count))


# ─── AI links search ─────────────────────────────────────────────────────────


class AIWebLinksSearchToolInput(BaseModel):
    prompt: str = Field(min_length=1, description="What to find web links for.")
    count: Optional[int] = Field(
        default=None,
        ge=10,
        le=200,
        description="Number of links to return (10-200). The API default is 10.",
    )


class AIWebLinksSearchTool(BaseTool):
    name: str = "ai_web_links_search_tool"
    description: str = (
        "Use Desearch AI to find web links relevant to a prompt. Returns "
        "search_results, a list of {title, link, snippet}, without a written "
        "summary. To keep results to one site, put site:<domain> in the prompt."
    )
    args_schema: ArgsSchema = AIWebLinksSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        prompt: str,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        # Web Search is the only tool this route still runs; the retired ones
        # are accepted by the API but dropped.
        return _call_client(
            "ai_web_links_search", **_compact(prompt=prompt, tools=["web"], count=count)
        )

    async def _arun(
        self,
        prompt: str,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "ai_web_links_search", **_compact(prompt=prompt, tools=["web"], count=count)
        )


class AIXLinksSearchToolInput(BaseModel):
    prompt: str = Field(min_length=1, description="What to find X posts about.")
    count: Optional[int] = Field(
        default=None,
        ge=10,
        le=200,
        description="Number of posts to return (10-200). The API default is 10.",
    )


class AIXLinksSearchTool(BaseTool):
    name: str = "ai_x_links_search_tool"
    description: str = (
        "Use Desearch AI to find X (Twitter) posts relevant to a prompt. Returns "
        "miner_tweets, a list of full posts (text, url, author, counts, date), "
        "without a written summary."
    )
    args_schema: ArgsSchema = AIXLinksSearchToolInput
    return_direct: bool = True

    def _run(
        self,
        prompt: str,
        count: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client("ai_x_links_search", **_compact(prompt=prompt, count=count))

    async def _arun(
        self,
        prompt: str,
        count: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "ai_x_links_search", **_compact(prompt=prompt, count=count)
        )


# ─── Web extract and Search API ──────────────────────────────────────────────


class WebExtractToolInput(BaseModel):
    url: str = Field(
        min_length=1, description="Public URL to read, for example https://desearch.ai."
    )
    format: Literal["text", "html"] = Field(
        default="text",
        description="'text' (the default) returns plain text; 'html' returns the HTML.",
    )
    js: bool = Field(
        default=False,
        description=(
            "Render the page in a headless browser first, for single-page apps "
            "and pages filled in by JavaScript. Billed at a separate tier."
        ),
    )
    wait: Optional[int] = Field(
        default=None,
        ge=0,
        le=30000,
        description="Extra milliseconds to wait after the page loads (0-30000). Needs js=true.",
    )

    @model_validator(mode="after")
    def check_wait_needs_js(self) -> "WebExtractToolInput":
        if self.wait is not None and not self.js:
            raise ValueError(
                "wait only applies when js is true. Set js=true or leave wait out."
            )
        return self


class WebExtractTool(BaseTool):
    name: str = "web_extract_tool"
    description: str = (
        "Fetch a public web page and return its content as plain text (default) "
        "or HTML. Set js=true for pages that need JavaScript to show their content."
    )
    args_schema: ArgsSchema = WebExtractToolInput
    return_direct: bool = True

    def _run(
        self,
        url: str,
        format: str = "text",
        js: bool = False,
        wait: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "extract", **_compact(url=url, format=format, js=js, wait=wait)
        )

    async def _arun(
        self,
        url: str,
        format: str = "text",
        js: bool = False,
        wait: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "extract", **_compact(url=url, format=format, js=js, wait=wait)
        )


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_day(name: str, value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    if not _ISO_DATE.match(value):
        raise ValueError(f"{name} must be a date in YYYY-MM-DD format, got {value!r}.")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} {value!r} is not a valid date.") from None


class SearchApiToolInput(BaseModel):
    query: str = Field(
        min_length=1, max_length=2000, description="What to search the web for."
    )
    mode: Optional[Literal["fast", "balanced"]] = Field(
        default=None,
        description="'fast' returns results sooner; 'balanced' (the API default) favors relevance.",
    )
    count: Optional[Literal[10, 15, 20]] = Field(
        default=None, description="Number of results: 10 (the API default), 15 or 20."
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Only pages published on or after this date (YYYY-MM-DD).",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Only pages published on or before this date (YYYY-MM-DD).",
    )
    highlights: Optional[bool] = Field(
        default=None,
        description="Include the passage that best matches the query (the API default is true).",
    )
    page_text: Optional[bool] = Field(
        default=None,
        description=(
            "Include the full text of each page (the API default is false). "
            "Responses get much larger."
        ),
    )

    @model_validator(mode="after")
    def check_dates(self) -> "SearchApiToolInput":
        start = _parse_day("start_date", self.start_date)
        end = _parse_day("end_date", self.end_date)
        if start and end and start > end:
            raise ValueError("start_date must be on or before end_date.")
        return self


class SearchApiTool(BaseTool):
    name: str = "search_api_tool"
    description: str = (
        "Search the web and get ranked pages (url, title, published_date), each "
        "with the passage that best matches the query. Filter by publish date; "
        "set page_text=true for the full page text."
    )
    args_schema: ArgsSchema = SearchApiToolInput
    return_direct: bool = True

    def _run(
        self,
        query: str,
        mode: Optional[str] = None,
        count: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        highlights: Optional[bool] = None,
        page_text: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_client(
            "search_api",
            **_compact(
                query=query,
                mode=mode,
                count=count,
                start_date=start_date,
                end_date=end_date,
                highlights=highlights,
                page_text=page_text,
            ),
        )

    async def _arun(
        self,
        query: str,
        mode: Optional[str] = None,
        count: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        highlights: Optional[bool] = None,
        page_text: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_client(
            "search_api",
            **_compact(
                query=query,
                mode=mode,
                count=count,
                start_date=start_date,
                end_date=end_date,
                highlights=highlights,
                page_text=page_text,
            ),
        )


# ─── TikTok, Facebook and Instagram ──────────────────────────────────────────
#
# One tool per platform with an ``action`` field. social.py maps each action to
# its desearch-py method and rejects fields the action does not use. List
# actions return {"results": [...], "next_cursor": ...}; the cursor comes from
# the x-desearch-next-cursor header through include_metadata=True.

_SOCIAL_COUNT = (
    "List actions only: maximum number of items (1-100). The API default is 10 "
    "for search and 20 otherwise. A page can hold fewer items without being the "
    "last one."
)
_SOCIAL_CURSOR = (
    "next_cursor from the previous page of the same action and target, sent "
    "back unchanged. Omit it for the first page. Never build or edit a cursor."
)
_SOCIAL_PAGING = (
    "Each action uses only its own fields. List actions return "
    "{results, next_cursor}: while next_cursor is set, call again with the same "
    "action and target and cursor=next_cursor to read the next page; it is null "
    "on the last page."
)


def _social_fields(model: BaseModel) -> Dict[str, Any]:
    return {
        name: getattr(model, name)
        for name in type(model).model_fields
        if name != "action"
    }


def _call_social(tool_name: str, platform: str, action: str, **fields: Any) -> Any:
    request = social_request(platform, action, **fields)
    if not request.returns_list:
        return _call_client(request.method, **request.kwargs)
    result = _call_client(request.method, include_metadata=True, **request.kwargs)
    return social_page(result, tool_name)


async def _acall_social(
    tool_name: str, platform: str, action: str, **fields: Any
) -> Any:
    request = social_request(platform, action, **fields)
    if not request.returns_list:
        return await _acall_client(request.method, **request.kwargs)
    result = await _acall_client(
        request.method, include_metadata=True, **request.kwargs
    )
    return social_page(result, tool_name)


class TikTokToolInput(BaseModel):
    action: Literal[
        "search", "post", "profile", "profile_posts", "comments", "hashtag", "trending"
    ] = Field(
        description=(
            "search: posts matching query. post: one post by id. profile: a user "
            "by username. profile_posts: a user's posts by username. comments: "
            "comments on a video by video_id. hashtag: posts for tag. trending: "
            "trending posts for region."
        )
    )
    query: Optional[str] = Field(
        default=None, description="action=search: keyword or phrase."
    )
    sort: Optional[str] = Field(
        default=None, description="action=search: sort option, when TikTok supports it."
    )
    id: Optional[str] = Field(default=None, description="action=post: TikTok video ID.")
    username: Optional[str] = Field(
        default=None,
        description="action=profile or profile_posts: TikTok username, without @.",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="action=comments: TikTok video ID (the id of a post).",
    )
    tag: Optional[str] = Field(
        default=None, description="action=hashtag: hashtag, without #."
    )
    region: Optional[str] = Field(
        default=None,
        description="action=trending: two-letter region code. The API default is US.",
    )
    count: Optional[int] = Field(default=None, ge=1, le=100, description=_SOCIAL_COUNT)
    cursor: Optional[str] = Field(
        default=None,
        description=(
            f"action=search, profile_posts, comments or hashtag: {_SOCIAL_CURSOR}"
        ),
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "TikTokToolInput":
        social_request("tiktok", self.action, **_social_fields(self))
        return self


class TikTokTool(BaseTool):
    name: str = "tiktok_tool"
    description: str = (
        "Read public TikTok data. Set action to 'search' (posts matching query), "
        "'post' (one post by video id), 'profile' (a user by username), "
        "'profile_posts' (a user's posts by username), 'comments' (comments on a "
        "video by video_id, which is a post's id), 'hashtag' (posts for tag) or "
        "'trending' (trending posts for a two-letter region, default US; one "
        f"page, no cursor). {_SOCIAL_PAGING}"
    )
    args_schema: ArgsSchema = TikTokToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        sort: Optional[str] = None,
        id: Optional[str] = None,
        username: Optional[str] = None,
        video_id: Optional[str] = None,
        tag: Optional[str] = None,
        region: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "tiktok",
            action,
            query=query,
            sort=sort,
            id=id,
            username=username,
            video_id=video_id,
            tag=tag,
            region=region,
            count=count,
            cursor=cursor,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        sort: Optional[str] = None,
        id: Optional[str] = None,
        username: Optional[str] = None,
        video_id: Optional[str] = None,
        tag: Optional[str] = None,
        region: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "tiktok",
            action,
            query=query,
            sort=sort,
            id=id,
            username=username,
            video_id=video_id,
            tag=tag,
            region=region,
            count=count,
            cursor=cursor,
        )


class FacebookToolInput(BaseModel):
    action: Literal[
        "search", "post", "profile", "profile_posts", "comments", "hashtag"
    ] = Field(
        description=(
            "search: Meta Ad Library ads matching query. post: one post by id. "
            "profile: a profile or page by profile_id. profile_posts: its posts "
            "by profile_id. comments: comments on a post by post_id. hashtag: "
            "Meta Ad Library ads for tag."
        )
    )
    query: Optional[str] = Field(
        default=None,
        description="action=search: Ad Library keyword, phrase or hashtag.",
    )
    sort: Optional[Literal["impressions", "recent"]] = Field(
        default=None,
        description=(
            "action=search: ad order. impressions (the API default) puts the most seen "
            "ads first; recent puts recently started ads first."
        ),
    )
    id: Optional[str] = Field(
        default=None, description="action=post: Facebook public post or media ID."
    )
    profile_id: Optional[str] = Field(
        default=None,
        description=(
            "action=profile or profile_posts: profile or page ID, username or "
            "public handle."
        ),
    )
    post_id: Optional[str] = Field(
        default=None, description="action=comments: Facebook public post or media ID."
    )
    tag: Optional[str] = Field(
        default=None, description="action=hashtag: hashtag or keyword, without #."
    )
    count: Optional[int] = Field(default=None, ge=1, le=100, description=_SOCIAL_COUNT)
    cursor: Optional[str] = Field(
        default=None,
        description=(
            f"action=search, profile_posts, comments or hashtag: {_SOCIAL_CURSOR}"
        ),
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "FacebookToolInput":
        social_request("facebook", self.action, **_social_fields(self))
        return self


class FacebookTool(BaseTool):
    name: str = "facebook_tool"
    description: str = (
        "Read public Facebook data. Set action to 'search' (Meta Ad Library ads "
        "matching query, not organic posts; sort impressions or recent), 'post' (one "
        "post by id), 'profile' (a profile or page by profile_id: ID, username or "
        "handle), 'profile_posts' (a profile's or page's posts by profile_id), 'comments' "
        "(comments on a post by post_id; about 10 per page) or 'hashtag' (Meta "
        f"Ad Library ads for tag). {_SOCIAL_PAGING}"
    )
    args_schema: ArgsSchema = FacebookToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        sort: Optional[str] = None,
        id: Optional[str] = None,
        profile_id: Optional[str] = None,
        post_id: Optional[str] = None,
        tag: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "facebook",
            action,
            query=query,
            sort=sort,
            id=id,
            profile_id=profile_id,
            post_id=post_id,
            tag=tag,
            count=count,
            cursor=cursor,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        sort: Optional[str] = None,
        id: Optional[str] = None,
        profile_id: Optional[str] = None,
        post_id: Optional[str] = None,
        tag: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "facebook",
            action,
            query=query,
            sort=sort,
            id=id,
            profile_id=profile_id,
            post_id=post_id,
            tag=tag,
            count=count,
            cursor=cursor,
        )


class InstagramToolInput(BaseModel):
    action: Literal[
        "search", "post", "profile", "profile_posts", "comments", "hashtag"
    ] = Field(
        description=(
            "search: profiles matching query. post: one post by id. profile: a "
            "user by username. profile_posts: a user's posts by username. "
            "comments: comments on a post by shortcode. hashtag: posts for tag."
        )
    )
    query: Optional[str] = Field(
        default=None, description="action=search: keyword, username or phrase."
    )
    id: Optional[str] = Field(
        default=None,
        description="action=post: post shortcode, media ID or public post URL.",
    )
    username: Optional[str] = Field(
        default=None,
        description="action=profile or profile_posts: Instagram username, without @.",
    )
    shortcode: Optional[str] = Field(
        default=None,
        description=(
            "action=comments: the post shortcode, the code in "
            "instagram.com/p/<shortcode>/. Numeric media IDs return 404."
        ),
    )
    tag: Optional[str] = Field(
        default=None, description="action=hashtag: hashtag, without #."
    )
    count: Optional[int] = Field(default=None, ge=1, le=100, description=_SOCIAL_COUNT)
    cursor: Optional[str] = Field(
        default=None,
        description=f"action=profile_posts, comments or hashtag: {_SOCIAL_CURSOR}",
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "InstagramToolInput":
        social_request("instagram", self.action, **_social_fields(self))
        return self


class InstagramTool(BaseTool):
    name: str = "instagram_tool"
    description: str = (
        "Read public Instagram data. Set action to 'search' (profiles matching "
        "query; one page, no cursor), 'post' (one post by id: shortcode, media ID "
        "or post URL), 'profile' (a user by username), 'profile_posts' (a user's "
        "posts by username), 'comments' (comments on a post by shortcode, the "
        "code in instagram.com/p/<shortcode>/) or 'hashtag' (posts for tag). "
        f"{_SOCIAL_PAGING}"
    )
    args_schema: ArgsSchema = InstagramToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        id: Optional[str] = None,
        username: Optional[str] = None,
        shortcode: Optional[str] = None,
        tag: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "instagram",
            action,
            query=query,
            id=id,
            username=username,
            shortcode=shortcode,
            tag=tag,
            count=count,
            cursor=cursor,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        id: Optional[str] = None,
        username: Optional[str] = None,
        shortcode: Optional[str] = None,
        tag: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "instagram",
            action,
            query=query,
            id=id,
            username=username,
            shortcode=shortcode,
            tag=tag,
            count=count,
            cursor=cursor,
        )


# ─── YouTube, Reddit and Threads ─────────────────────────────────────────────
#
# Same shape as the tools above, but these routes take no count: a page is
# whatever the platform returns. Threads never pages. Each call is billed as
# one request, including an empty page.


class YouTubeToolInput(BaseModel):
    action: Literal[
        "search", "video", "transcript", "comments", "channel", "channel_videos"
    ] = Field(
        description=(
            "search: videos, Shorts, channels or playlists matching query. video: "
            "one video by id. transcript: a video's captions by video_id. "
            "comments: top-level comments on a video by video_id. channel: a "
            "channel by handle. channel_videos: a channel's uploads by handle."
        )
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            'action=search: keyword or phrase. Use intitle:"..." for stricter '
            "title matching."
        ),
    )
    type: Optional[Literal["videos", "shorts", "channels", "playlists"]] = Field(
        default=None,
        description="action=search: only this kind of result. Omit for mixed results.",
    )
    upload_date: Optional[
        Literal["today", "this_week", "this_month", "this_year"]
    ] = Field(default=None, description="action=search: only videos uploaded in this window.")
    sort_by: Optional[Literal["relevance", "popular"]] = Field(
        default=None, description="action=search: result order."
    )
    duration: Optional[
        Literal["under_3_min", "between_3_and_20_min", "over_20_min"]
    ] = Field(default=None, description="action=search: video length (not Shorts).")
    id: Optional[str] = Field(
        default=None, description="action=video: video or Short ID, or its URL."
    )
    video_id: Optional[str] = Field(
        default=None,
        description="action=transcript or comments: video or Short ID, or its URL.",
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "action=transcript: caption language code such as en or es. Omit for "
            "the default track. A video without captions returns a not-found error."
        ),
    )
    order: Optional[Literal["top", "newest"]] = Field(
        default=None, description="action=comments: comment order."
    )
    handle: Optional[str] = Field(
        default=None,
        description=(
            "action=channel or channel_videos: channel handle (with or without @) "
            "or channel ID (UC...)."
        ),
    )
    sort: Optional[Literal["latest", "popular"]] = Field(
        default=None,
        description="action=channel_videos: video order. The API default is latest.",
    )
    cursor: Optional[str] = Field(
        default=None,
        description=f"action=search, comments or channel_videos: {_SOCIAL_CURSOR}",
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "YouTubeToolInput":
        social_request("youtube", self.action, **_social_fields(self))
        return self


class YouTubeTool(BaseTool):
    name: str = "youtube_tool"
    description: str = (
        "Read public YouTube data. Set action to 'search' (videos, Shorts, "
        "channels or playlists matching query; filters type, upload_date, "
        "sort_by, duration), 'video' (one video by id or URL: views, likes, "
        "chapters, caption languages), 'transcript' (a video's captions by "
        "video_id, optional language), 'comments' (top-level comments by "
        "video_id, order top or newest), 'channel' (a channel by handle) or "
        f"'channel_videos' (a channel's uploads by handle, sort latest or "
        f"popular). {_SOCIAL_PAGING}"
    )
    args_schema: ArgsSchema = YouTubeToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        type: Optional[str] = None,
        upload_date: Optional[str] = None,
        sort_by: Optional[str] = None,
        duration: Optional[str] = None,
        id: Optional[str] = None,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        order: Optional[str] = None,
        handle: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "youtube",
            action,
            query=query,
            type=type,
            upload_date=upload_date,
            sort_by=sort_by,
            duration=duration,
            id=id,
            video_id=video_id,
            language=language,
            order=order,
            handle=handle,
            sort=sort,
            cursor=cursor,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        type: Optional[str] = None,
        upload_date: Optional[str] = None,
        sort_by: Optional[str] = None,
        duration: Optional[str] = None,
        id: Optional[str] = None,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        order: Optional[str] = None,
        handle: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "youtube",
            action,
            query=query,
            type=type,
            upload_date=upload_date,
            sort_by=sort_by,
            duration=duration,
            id=id,
            video_id=video_id,
            language=language,
            order=order,
            handle=handle,
            sort=sort,
            cursor=cursor,
        )


class RedditToolInput(BaseModel):
    action: Literal["search", "subreddit", "subreddit_posts", "post", "comments"] = (
        Field(
            description=(
                "search: posts across Reddit matching query. subreddit: a "
                "subreddit by name. subreddit_posts: a subreddit's posts by name. "
                "post: one post by id. comments: a post's comment tree by post_id."
            )
        )
    )
    query: Optional[str] = Field(
        default=None, description="action=search: keyword or phrase."
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "action=subreddit or subreddit_posts: subreddit name without r/. It "
            "is case-sensitive (AskReddit, not askreddit)."
        ),
    )
    sort: Optional[str] = Field(
        default=None,
        description=(
            "action=search: relevance, new, top or comment_count. "
            "action=subreddit_posts: best, hot, new, top or rising."
        ),
    )
    timeframe: Optional[Literal["all", "day", "week", "month", "year"]] = Field(
        default=None,
        description=(
            "action=search or subreddit_posts: time window. On subreddit_posts it "
            "applies to sort=top; without a sort it means top."
        ),
    )
    id: Optional[str] = Field(
        default=None,
        description="action=post: post ID (such as 1q6pxwn, with or without t3_) or URL.",
    )
    post_id: Optional[str] = Field(
        default=None,
        description="action=comments: post ID (with or without t3_) or URL.",
    )
    cursor: Optional[str] = Field(
        default=None,
        description=f"action=search, subreddit_posts or comments: {_SOCIAL_CURSOR}",
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "RedditToolInput":
        social_request("reddit", self.action, **_social_fields(self))
        return self


class RedditTool(BaseTool):
    name: str = "reddit_tool"
    description: str = (
        "Read public Reddit data. Set action to 'search' (posts across all "
        "subreddits matching query), 'subreddit' (a subreddit by name: rules, "
        "weekly active users), 'subreddit_posts' (a subreddit's posts by name, "
        "with sort and timeframe), 'post' (one post by id or URL, without "
        "comments) or 'comments' (a post's comments by post_id, as a tree with "
        f"the replies Reddit loaded). {_SOCIAL_PAGING}"
    )
    args_schema: ArgsSchema = RedditToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        name: Optional[str] = None,
        sort: Optional[str] = None,
        timeframe: Optional[str] = None,
        id: Optional[str] = None,
        post_id: Optional[str] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "reddit",
            action,
            query=query,
            name=name,
            sort=sort,
            timeframe=timeframe,
            id=id,
            post_id=post_id,
            cursor=cursor,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        name: Optional[str] = None,
        sort: Optional[str] = None,
        timeframe: Optional[str] = None,
        id: Optional[str] = None,
        post_id: Optional[str] = None,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "reddit",
            action,
            query=query,
            name=name,
            sort=sort,
            timeframe=timeframe,
            id=id,
            post_id=post_id,
            cursor=cursor,
        )


class ThreadsToolInput(BaseModel):
    action: Literal["search", "search_users", "profile", "profile_posts", "post"] = (
        Field(
            description=(
                "search: posts matching query. search_users: accounts matching "
                "query. profile: a user by username. profile_posts: a user's "
                "latest posts by username. post: one post with its replies by id."
            )
        )
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "action=search: keyword or phrase. action=search_users: username or name."
        ),
    )
    start_date: Optional[str] = Field(
        default=None,
        description="action=search: only posts on or after this date (YYYY-MM-DD).",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="action=search: only posts on or before this date (YYYY-MM-DD).",
    )
    username: Optional[str] = Field(
        default=None,
        description="action=profile or profile_posts: Threads username, without @.",
    )
    id: Optional[str] = Field(
        default=None, description="action=post: post URL or shortcode."
    )

    @model_validator(mode="after")
    def check_action_fields(self) -> "ThreadsToolInput":
        social_request("threads", self.action, **_social_fields(self))
        return self


class ThreadsTool(BaseTool):
    name: str = "threads_tool"
    description: str = (
        "Read public Threads data. Set action to 'search' (posts matching query, "
        "optional start_date and end_date), 'search_users' (accounts matching "
        "query), 'profile' (a user by username), 'profile_posts' (a user's latest "
        "public posts) or 'post' (one post by URL or shortcode, with its replies "
        "and the author's thread continuation). Each action uses only its own "
        "fields. List actions return {results, next_cursor}; Threads returns one "
        "page, so next_cursor is always null."
    )
    args_schema: ArgsSchema = ThreadsToolInput
    return_direct: bool = True

    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        username: Optional[str] = None,
        id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_social(
            self.name,
            "threads",
            action,
            query=query,
            start_date=start_date,
            end_date=end_date,
            username=username,
            id=id,
        )

    async def _arun(
        self,
        action: str,
        query: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        username: Optional[str] = None,
        id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_social(
            self.name,
            "threads",
            action,
            query=query,
            start_date=start_date,
            end_date=end_date,
            username=username,
            id=id,
        )


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


# ─── Company Search ──────────────────────────────────────────────────────────
#
# desearch-py ships Company Search in 1.8.0, which is not on PyPI yet, so these
# tools call the /desearch/companies routes over HTTP (see companies.py).

_CompanyRole = Literal[
    "ceo",
    "founder",
    "cto",
    "cfo",
    "coo",
    "cmo",
    "president",
    "owner",
    "managing_director",
]
_CompanyRequirement = Literal["email", "phone", "linkedin", "leader", "leader_email"]
_CompanyTaskStatus = Literal[
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
]

_COMPANY_SEARCH_TOOL = "company_search_tool"
_COMPANY_ENRICH_TOOL = "company_enrich_tool"
_COMPANY_TASKS_TOOL = "company_tasks_tool"
_COMPANY_CONTROL_TOOL = "company_task_control_tool"

_ROLES_DESCRIPTION = (
    "Leaders to find: ceo, founder, cto, cfo, coo, cmo, president, owner, "
    "managing_director. Default ceo, founder, cto, cfo."
)
_INFER_EMAILS_DESCRIPTION = (
    "Guess leaders' emails from the company's address pattern (default true). "
    "Inferred emails say source 'inferred' and have a confidence."
)
_VERIFY_EMAILS_DESCRIPTION = "Check every email: syntax, mail server and web (default true)."
_METADATA_DESCRIPTION = (
    "Your own labels for the task, for example {'campaign': 'q4-outbound'}: up to 10 "
    "keys of 1-40 characters (a-z, 0-9, '_', '.', '-'), string values up to 200 "
    "characters. They do not change what is found or what it costs."
)
_IDEMPOTENCY_DESCRIPTION = (
    "Any unique string. Sending the same key again returns the first task "
    "instead of starting and billing a new one."
)
_TIMEOUT_DESCRIPTION = (
    "Seconds to wait for the task to finish (default 300). If it is still "
    "running, the companies delivered so far are returned with the task_id."
)
_TASK_ID_DESCRIPTION = (
    "task_id of a Company Search or Company Enrich task started earlier. Waits on "
    "that task and returns its companies instead of starting a new one. Free."
)


async def _invoke_companies(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    async with CompanyClient(api_key, base_url) as client:
        return await operation(client, **kwargs)


def _call_companies(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return _run_coroutine(
            lambda: _invoke_companies(api_key, base_url, operation, **kwargs)
        )
    except Exception as exc:
        return _people_error(exc)


async def _acall_companies(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return await _invoke_companies(api_key, base_url, operation, **kwargs)
    except Exception as exc:
        return _people_error(exc)


class _CompanyTaskTool(BaseTool):
    """Shared start-or-wait logic of company_search_tool and company_enrich_tool."""

    return_direct: bool = True
    base_url: str = COMPANY_API_BASE_URL
    poll_interval: float = 5.0

    def _task_kwargs(
        self,
        kind: str,
        body: Optional[Dict[str, Any]],
        idempotency_key: Optional[str],
        timeout: int,
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "tool_name": self.name,
            "control_tool_name": _COMPANY_CONTROL_TOOL,
            "timeout": timeout,
            "poll_interval": self.poll_interval,
        }
        if task_id:
            return {**kwargs, "task_id": task_id}
        return {
            **kwargs,
            "kind": kind,
            "body": body,
            "idempotency_key": idempotency_key,
        }


class CompanySearchToolInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=1000,
        description=(
            "The companies to find, in plain language: what they do, where, how far "
            "along. For example 'Series A fintech startups in Berlin building payment "
            "infrastructure'. Required to start a search."
        ),
    )
    count: int = Field(
        default=25,
        ge=5,
        le=200,
        description=(
            "How many companies to deliver (5-200). The task reserves this many at the "
            "Company Search price and bills only the companies it delivers."
        ),
    )
    min_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Only deliver companies with at least this fit score (0-100).",
    )
    exclude: Optional[List[str]] = Field(
        default=None,
        max_length=1000,
        description=(
            "Companies to leave out, as domains or URLs, for example ones you already "
            "contacted (up to 1000)."
        ),
    )
    countries: Optional[List[str]] = Field(
        default=None,
        max_length=20,
        description=(
            "Where the companies should be: ISO codes or country names, for example "
            "['DE', 'Austria'] (up to 20)."
        ),
    )
    require: Optional[List[_CompanyRequirement]] = Field(
        default=None,
        description=(
            "Only deliver companies that have all of these: email, phone, linkedin (the "
            "company's LinkedIn page), leader (one of roles), leader_email (a leader with "
            "an email). A company without them is not delivered or billed."
        ),
    )
    roles: Optional[List[_CompanyRole]] = Field(
        default=None, description=_ROLES_DESCRIPTION
    )
    infer_emails: Optional[bool] = Field(
        default=None, description=_INFER_EMAILS_DESCRIPTION
    )
    verify_emails: Optional[bool] = Field(
        default=None, description=_VERIFY_EMAILS_DESCRIPTION
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description=_METADATA_DESCRIPTION
    )
    idempotency_key: Optional[str] = Field(
        default=None, max_length=200, description=_IDEMPOTENCY_DESCRIPTION
    )
    timeout: int = Field(default=300, ge=0, le=1800, description=_TIMEOUT_DESCRIPTION)
    task_id: Optional[str] = Field(default=None, description=_TASK_ID_DESCRIPTION)

    @model_validator(mode="after")
    def check_search_or_task(self) -> "CompanySearchToolInput":
        if self.task_id:
            return self
        if not self.query:
            raise ValueError(
                "Set query to start a Company Search, or task_id to keep waiting on one."
            )
        # Refuse values the API would reject before anything is billed.
        company_search_body(
            self.query,
            countries=self.countries,
            require=self.require,
            roles=self.roles,
            metadata=self.metadata,
        )
        return self


class CompanySearchTool(_CompanyTaskTool):
    name: str = _COMPANY_SEARCH_TOOL
    description: str = (
        "Finds companies on the web that match a plain-language description (for "
        "example 'Series A fintech startups in Berlin building payment infrastructure') "
        "and returns compact companies: domain, name, url, location, fit score and why "
        "it matched, primary email and contact, public emails, phones and social "
        "profiles with the page each came from, and leaders (CEO, founders, CTO, CFO by "
        "default) with their LinkedIn profiles and emails. BILLED: starting a search "
        "reserves `count` companies at the Company Search price; you pay only for the "
        "companies it delivers (a minimum of 2 when it delivers any), and a search "
        "that delivers none is free. SLOW: it runs in the background for minutes. This "
        "tool waits up to `timeout` seconds, then returns the companies delivered so "
        "far and the task_id. To keep waiting, call it again with only task_id (free; "
        "it does not start or bill a new search). For domains you already have, use "
        "company_enrich_tool."
    )
    args_schema: ArgsSchema = CompanySearchToolInput

    def _search_kwargs(
        self,
        query: Optional[str],
        count: int,
        min_score: Optional[int],
        exclude: Optional[List[str]],
        countries: Optional[List[str]],
        require: Optional[List[str]],
        roles: Optional[List[str]],
        infer_emails: Optional[bool],
        verify_emails: Optional[bool],
        metadata: Optional[Dict[str, str]],
        idempotency_key: Optional[str],
        timeout: int,
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        body = None
        if not task_id:
            if not query:
                raise ValueError(
                    "Set query to start a Company Search, or task_id to keep waiting on one."
                )
            body = company_search_body(
                query,
                count=count,
                min_score=min_score,
                exclude=exclude,
                countries=countries,
                require=require,
                roles=roles,
                infer_emails=infer_emails,
                verify_emails=verify_emails,
                metadata=metadata,
            )
        return self._task_kwargs("search", body, idempotency_key, timeout, task_id)

    def _run(
        self,
        query: Optional[str] = None,
        count: int = 25,
        min_score: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        require: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        infer_emails: Optional[bool] = None,
        verify_emails: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        kwargs = self._search_kwargs(
            query,
            count,
            min_score,
            exclude,
            countries,
            require,
            roles,
            infer_emails,
            verify_emails,
            metadata,
            idempotency_key,
            timeout,
            task_id,
        )
        return _call_companies(_require_api_key(), self.base_url, run_company_task, **kwargs)

    async def _arun(
        self,
        query: Optional[str] = None,
        count: int = 25,
        min_score: Optional[int] = None,
        exclude: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        require: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        infer_emails: Optional[bool] = None,
        verify_emails: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        kwargs = self._search_kwargs(
            query,
            count,
            min_score,
            exclude,
            countries,
            require,
            roles,
            infer_emails,
            verify_emails,
            metadata,
            idempotency_key,
            timeout,
            task_id,
        )
        return await _acall_companies(
            _require_api_key(), self.base_url, run_company_task, **kwargs
        )


class CompanyEnrichToolInput(BaseModel):
    domains: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "1-200 company domains or URLs, for example ['acme.com', "
            "'https://www.beta.io/about']. Each is reduced to its host without www.; "
            "duplicates are removed. Required to start an enrichment."
        ),
    )
    roles: Optional[List[_CompanyRole]] = Field(
        default=None, description=_ROLES_DESCRIPTION
    )
    infer_emails: Optional[bool] = Field(
        default=None, description=_INFER_EMAILS_DESCRIPTION
    )
    verify_emails: Optional[bool] = Field(
        default=None, description=_VERIFY_EMAILS_DESCRIPTION
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description=_METADATA_DESCRIPTION
    )
    idempotency_key: Optional[str] = Field(
        default=None, max_length=200, description=_IDEMPOTENCY_DESCRIPTION
    )
    timeout: int = Field(default=300, ge=0, le=1800, description=_TIMEOUT_DESCRIPTION)
    task_id: Optional[str] = Field(default=None, description=_TASK_ID_DESCRIPTION)

    @model_validator(mode="after")
    def check_domains_or_task(self) -> "CompanyEnrichToolInput":
        if self.task_id:
            return self
        if not self.domains:
            raise ValueError(
                "Set domains to start a Company Enrich, or task_id to keep waiting on one."
            )
        company_enrich_body(self.domains, roles=self.roles, metadata=self.metadata)
        return self


class CompanyEnrichTool(_CompanyTaskTool):
    name: str = _COMPANY_ENRICH_TOOL
    description: str = (
        "Collects public contact data for companies you already know, by domain: one "
        "company per domain with name, url, location, primary email and contact, public "
        "emails, phones and social profiles with the page each came from, and leaders "
        "(CEO, founders, CTO, CFO by default) with their LinkedIn profiles and emails. "
        "No fit score (score is null). BILLED: starting it reserves one company per "
        "domain at the Company Enrich price; you pay only for the companies it delivers "
        "(a minimum of 1), and a task that delivers none is free. SLOW: it runs in the "
        "background for minutes. This tool waits up to `timeout` seconds, then returns "
        "the companies delivered so far and the task_id. To keep waiting, call it again "
        "with only task_id (free; it does not start or bill a new task). To find "
        "companies from a description, use company_search_tool."
    )
    args_schema: ArgsSchema = CompanyEnrichToolInput

    def _enrich_kwargs(
        self,
        domains: Optional[List[str]],
        roles: Optional[List[str]],
        infer_emails: Optional[bool],
        verify_emails: Optional[bool],
        metadata: Optional[Dict[str, str]],
        idempotency_key: Optional[str],
        timeout: int,
        task_id: Optional[str],
    ) -> Dict[str, Any]:
        body = None
        if not task_id:
            if not domains:
                raise ValueError(
                    "Set domains to start a Company Enrich, or task_id to keep waiting on one."
                )
            body = company_enrich_body(
                domains,
                roles=roles,
                infer_emails=infer_emails,
                verify_emails=verify_emails,
                metadata=metadata,
            )
        return self._task_kwargs("enrich", body, idempotency_key, timeout, task_id)

    def _run(
        self,
        domains: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        infer_emails: Optional[bool] = None,
        verify_emails: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        kwargs = self._enrich_kwargs(
            domains,
            roles,
            infer_emails,
            verify_emails,
            metadata,
            idempotency_key,
            timeout,
            task_id,
        )
        return _call_companies(_require_api_key(), self.base_url, run_company_task, **kwargs)

    async def _arun(
        self,
        domains: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        infer_emails: Optional[bool] = None,
        verify_emails: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
        timeout: int = 300,
        task_id: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        kwargs = self._enrich_kwargs(
            domains,
            roles,
            infer_emails,
            verify_emails,
            metadata,
            idempotency_key,
            timeout,
            task_id,
        )
        return await _acall_companies(
            _require_api_key(), self.base_url, run_company_task, **kwargs
        )


class CompanyTasksToolInput(BaseModel):
    type: Optional[Literal["search", "enrich"]] = Field(
        default=None,
        description="Only Company Search ('search') or only Company Enrich ('enrich') tasks.",
    )
    status: Optional[List[_CompanyTaskStatus]] = Field(
        default=None,
        description="Only tasks in these statuses, for example ['running', 'paused'].",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Tasks per page (1-100).")
    cursor: Optional[str] = Field(
        default=None,
        description="next_cursor from the previous page. Keep the same filters when you page.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Only tasks whose metadata has all of these labels (exact match, up to 3), "
            "for example {'campaign': 'q4-outbound'}."
        ),
    )

    @model_validator(mode="after")
    def check_metadata_filters(self) -> "CompanyTasksToolInput":
        if self.metadata and len(self.metadata) > 3:
            raise ValueError("Filter by at most 3 metadata labels.")
        return self


class CompanyTasksTool(BaseTool):
    name: str = _COMPANY_TASKS_TOOL
    description: str = (
        "Lists your Desearch Company Search and Company Enrich tasks, newest first, "
        "with status, counts, cost, why each ended task stopped (completion_reason, "
        "coverage_limited), warnings and metadata labels. Filter by type (search or "
        "enrich), status and metadata. Use it to find a task_id, then read the task's "
        "companies with company_search_tool or company_enrich_tool and only task_id. "
        "Free and instant: it never starts a task. People Search tasks are not listed."
    )
    args_schema: ArgsSchema = CompanyTasksToolInput
    return_direct: bool = True
    base_url: str = COMPANY_API_BASE_URL

    def _run(
        self,
        type: Optional[str] = None,
        status: Optional[List[str]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_companies(
            _require_api_key(),
            self.base_url,
            list_company_tasks,
            tool_name=self.name,
            type=type,
            status=status,
            limit=limit,
            cursor=cursor,
            metadata=metadata,
        )

    async def _arun(
        self,
        type: Optional[str] = None,
        status: Optional[List[str]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_companies(
            _require_api_key(),
            self.base_url,
            list_company_tasks,
            tool_name=self.name,
            type=type,
            status=status,
            limit=limit,
            cursor=cursor,
            metadata=metadata,
        )


class CompanyTaskControlToolInput(BaseModel):
    task_id: str = Field(
        min_length=1,
        description="task_id of a Company Search or Company Enrich task.",
    )
    action: Literal["pause", "resume", "cancel"] = Field(
        description=(
            "'pause' (a running task finishes its current batch first), 'resume' (no new "
            "charge) or 'cancel' (stops it for good; cannot be undone)."
        )
    )


class CompanyTaskControlTool(BaseTool):
    name: str = _COMPANY_CONTROL_TOOL
    description: str = (
        "Pauses, resumes or cancels a Desearch Company Search or Company Enrich task "
        "and returns its updated status. Free: it charges and refunds nothing by itself. "
        "Pause keeps the companies delivered so far and the reservation; a task paused "
        "for more than 7 days is cancelled. Resume continues where the task stopped with "
        "no new charge. Cancel stops it for good: you pay only for the companies it "
        "already delivered and the rest of the reservation is returned. Cancel cannot be "
        "undone."
    )
    args_schema: ArgsSchema = CompanyTaskControlToolInput
    return_direct: bool = True
    base_url: str = COMPANY_API_BASE_URL

    def _run(
        self,
        task_id: str,
        action: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return _call_companies(
            _require_api_key(),
            self.base_url,
            control_company_task,
            task_id=task_id,
            action=action,
            tool_name=_COMPANY_SEARCH_TOOL,
            control_tool_name=self.name,
            enrich_tool_name=_COMPANY_ENRICH_TOOL,
        )

    async def _arun(
        self,
        task_id: str,
        action: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await _acall_companies(
            _require_api_key(),
            self.base_url,
            control_company_task,
            task_id=task_id,
            action=action,
            tool_name=_COMPANY_SEARCH_TOOL,
            control_tool_name=self.name,
            enrich_tool_name=_COMPANY_ENRICH_TOOL,
        )


# Monitors. desearch-py ships them in 1.13.0, which is not on PyPI yet, so these
# tools call the /desearch/monitors routes over HTTP (see monitors.py).

_MonitorSource = Literal[
    "web", "x", "reddit", "youtube", "threads", "tiktok", "instagram", "facebook", "meta_ads"
]
_MonitorInterval = Literal["1h", "3h", "6h", "12h", "1d", "7d"]
_MonitorEvent = Literal["monitor.items.found", "monitor.run.completed", "monitor.run.failed"]

_MONITOR_BILLING = (
    "Creating, reading, changing and deleting monitors is free. Each run is billed: its "
    "searches at their list price, AI screening at $0.05 per 1,000 items screened, and the summary's LLM "
    "usage plus 50% (when summary is on). A run holds max_run_cost_usd on the balance and "
    "bills only what it used; a run with no new items still pays for its searches. A "
    "watched account is one feed read per run: an X account about $0.003 (a 20-post "
    "timeline), a subreddit, YouTube channel or Threads account $0.00207, TikTok $0.00207 "
    "per post (search, account or hashtag), an Instagram account or hashtag $0.02484 (up "
    "to 12 posts) and a "
    "Facebook page $0.0207 (up to 10 posts). Meta ads cost $0.0207 per 10 Ad Library "
    "records (meta_ads_count 10 is $0.0207, 30 is $0.0621)."
)
_MONITOR_ID_DESCRIPTION = "monitor_id returned when the monitor was created."
_MONITOR_QUERY_DESCRIPTION = (
    "What to watch, in plain words, for example 'Bittensor subnet launches and updates' "
    "(3-500 characters). Every source is searched with it."
)
_MONITOR_CRITERIA_DESCRIPTION = (
    "Up to 5 conditions an item must meet to be kept, each up to 300 characters, for "
    "example ['Announces a new subnet, release or partnership']. Each is one yes/no "
    "screening question."
)
_MONITOR_SOURCES_DESCRIPTION = (
    "Where to look: web, x, reddit, youtube, threads, tiktok, instagram, facebook, meta_ads "
    "(default web and x). A source with watched accounts or hashtags reads them instead of "
    "searching. instagram and facebook only read watched feeds, so instagram needs "
    "instagram_accounts or instagram_hashtags and facebook needs facebook_pages (HTTP 422 "
    "without). meta_ads searches the Meta Ad Library by keyword."
)
_MONITOR_INTERVAL_DESCRIPTION = "How often it runs: 1h, 3h, 6h, 12h, 1d or 7d (default 1d)."
_MONITOR_MIN_SCORE_DESCRIPTION = (
    "Relevance (0-100) an item needs to be kept (default 60). The score is the lowest of "
    "'is it about the query' and the criteria."
)
_MONITOR_SUMMARY_DESCRIPTION = (
    "Write a short digest of the items each run keeps (its LLM usage is billed plus 50%)."
)
_MONITOR_WEB_COUNT_DESCRIPTION = "Web pages read per run: 10, 15 or 20 (default 10)."
_MONITOR_X_COUNT_DESCRIPTION = "Newest X posts read per run, 1-100 (default 20)."
_MONITOR_X_LANG_DESCRIPTION = "Only X posts in this language, for example 'en'."
_MONITOR_X_MIN_LIKES_DESCRIPTION = "Only X posts with at least this many likes."
_MONITOR_ACCOUNTS_NOTE = (
    "Up to 10. With accounts the source reads their newest posts each run instead of "
    "searching by keyword; query and criteria still decide what is kept (min_score 0 "
    "keeps every new post). The API returns the names cleaned."
)
_MONITOR_X_ACCOUNTS_DESCRIPTION = (
    "X accounts to watch: 'OpenAI', '@OpenAI' or an x.com profile URL. x_count is not "
    "used with accounts (a timeline is 20 posts); x_lang and x_min_likes still filter. "
    f"{_MONITOR_ACCOUNTS_NOTE}"
)
_MONITOR_REDDIT_SUBREDDITS_DESCRIPTION = (
    f"Subreddits to watch: 'LocalLLaMA', 'r/LocalLLaMA' or a URL. {_MONITOR_ACCOUNTS_NOTE}"
)
_MONITOR_YOUTUBE_CHANNELS_DESCRIPTION = (
    "YouTube channels to watch: '@handle', a 'UC...' channel ID or a channel URL. "
    f"{_MONITOR_ACCOUNTS_NOTE}"
)
_MONITOR_TIKTOK_COUNT_DESCRIPTION = (
    "TikTok posts read per run, per account and per hashtag with tiktok_accounts or "
    "tiktok_hashtags: 10, 20 or 30 (default 10)."
)
_MONITOR_TIKTOK_SEARCH_NOTE = (
    "TikTok search has no working date filter and ranks by relevance, so a TikTok keyword "
    "monitor finds a new video only when it reaches TikTok's top results; to follow "
    "creators or a tag, watch their accounts (tiktok_accounts) or the hashtag "
    "(tiktok_hashtags)."
)
_MONITOR_TIKTOK_ACCOUNTS_DESCRIPTION = (
    f"TikTok accounts to watch: 'openai', '@openai' or a profile URL. {_MONITOR_ACCOUNTS_NOTE} "
    f"{_MONITOR_TIKTOK_SEARCH_NOTE}"
)
_MONITOR_HASHTAGS_NOTE = (
    "Up to 10, in any script. A hashtag feed lists popular as well as recent posts from "
    "any account, so it finds posts outside the watched accounts; with accounts or "
    "hashtags the source reads those feeds each run instead of searching by keyword; query "
    "and criteria still decide what is kept (min_score 0 keeps every new post). The API "
    "returns them without # (a tag URL becomes its tag). Items read from a hashtag have "
    "data.hashtag."
)
_MONITOR_TIKTOK_HASHTAGS_DESCRIPTION = (
    "TikTok hashtags to watch: 'skincare', '#სილამაზე' or a tiktok.com/tag/... URL; "
    f"tiktok_count posts per hashtag per run at $0.00207 each. {_MONITOR_HASHTAGS_NOTE}"
)
_MONITOR_INSTAGRAM_HASHTAGS_DESCRIPTION = (
    "Instagram hashtags to watch with the instagram source: 'skincare', '#სილამაზე' or an "
    "instagram.com/explore/tags/... URL; up to 12 posts per hashtag per run, $0.02484 per "
    f"hashtag. The instagram source needs these or instagram_accounts. {_MONITOR_HASHTAGS_NOTE}"
)
_MONITOR_THREADS_ACCOUNTS_DESCRIPTION = (
    "Threads accounts to watch: 'zuck', '@zuck' or a profile URL, $0.00207 per account per "
    "run. Threads keyword search returns loosely matched posts, so accounts work better. "
    f"{_MONITOR_ACCOUNTS_NOTE}"
)
_MONITOR_ACCOUNT_ONLY_NOTE = (
    "1-10, and required while the source is on: it has no keyword search for posts, so "
    "it reads only these (HTTP 422 without any, on create and on update). Each run reads "
    "their newest posts; query and criteria still decide what is kept (min_score 0 keeps "
    "every new post). The API returns the names cleaned."
)
_MONITOR_INSTAGRAM_ACCOUNTS_DESCRIPTION = (
    "Instagram accounts to watch with the instagram source: 'openai', '@openai' or a "
    "profile URL, up to 10. Up to 12 posts per account per run, $0.02484 per account. "
    "The instagram source has no keyword search for posts, so it reads only these and "
    "instagram_hashtags, and needs one of them (HTTP 422 without either, on create and on "
    "update). Each run reads their newest posts; query and criteria still decide what is "
    "kept (min_score 0 keeps every new post). The API returns the names cleaned."
)
_MONITOR_FACEBOOK_PAGES_DESCRIPTION = (
    "Facebook pages to watch with the facebook source: a page name ('NASA'), a numeric "
    "page ID or a page URL (profile.php?id=... and /pages/Name/123... become the ID). Up "
    f"to 10 posts per page per run, $0.0207 per page. {_MONITOR_ACCOUNT_ONLY_NOTE} Facebook "
    "has no organic hashtag feed; for ads by keyword, use the meta_ads source."
)
_MONITOR_META_ADS_NOTE = (
    "meta_ads searches the Meta Ad Library by keyword, so its query should be a brand, "
    "product or short phrase such as 'ChatGPT'; a sentence finds nothing. An ad is new the "
    "first time a run sees it if it started within the last 30 days."
)
_MONITOR_META_ADS_COUNT_DESCRIPTION = (
    "Meta Ad Library records read per run with the meta_ads source: 10, 20 or 30 (default "
    f"10; 10 is $0.0207, 30 is $0.0621). {_MONITOR_META_ADS_NOTE}"
)
_MONITOR_WEBHOOK_URL_DESCRIPTION = (
    "HTTPS URL on a public address that receives new items as signed POSTs. Redirects "
    "are not followed."
)
_MONITOR_WEBHOOK_EVENTS_DESCRIPTION = (
    "Webhook events: monitor.items.found (a run kept at least one item), "
    "monitor.run.completed (every completed run), monitor.run.failed. Default "
    "monitor.items.found and monitor.run.failed."
)
_MONITOR_METADATA_DESCRIPTION = (
    "Your own labels, for example {'team': 'growth'}: up to 20 keys of 1-64 characters "
    "with string values up to 500 characters. Echoed in webhooks; they do not change "
    "what is found."
)


async def _invoke_monitors(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    async with MonitorClient(api_key, base_url) as client:
        return await operation(client, **kwargs)


def _call_monitors(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return _run_coroutine(
            lambda: _invoke_monitors(api_key, base_url, operation, **kwargs)
        )
    except Exception as exc:
        return _people_error(exc)


async def _acall_monitors(
    api_key: str, base_url: str, operation: Callable[..., Awaitable[Any]], **kwargs: Any
) -> Any:
    try:
        return await _invoke_monitors(api_key, base_url, operation, **kwargs)
    except Exception as exc:
        return _people_error(exc)


class _MonitorTool(BaseTool):
    """Shared settings of the Monitors tools."""

    return_direct: bool = True
    base_url: str = MONITOR_API_BASE_URL

    def _call(self, operation: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        return _call_monitors(_require_api_key(), self.base_url, operation, **kwargs)

    async def _acall(self, operation: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
        return await _acall_monitors(_require_api_key(), self.base_url, operation, **kwargs)


class _MonitorOptionsInput(BaseModel):
    web_count: Optional[Literal[10, 15, 20]] = Field(
        default=None, description=_MONITOR_WEB_COUNT_DESCRIPTION
    )
    x_count: Optional[int] = Field(
        default=None, ge=1, le=100, description=_MONITOR_X_COUNT_DESCRIPTION
    )
    x_lang: Optional[str] = Field(
        default=None, min_length=2, max_length=3, description=_MONITOR_X_LANG_DESCRIPTION
    )
    x_min_likes: Optional[int] = Field(
        default=None, ge=0, description=_MONITOR_X_MIN_LIKES_DESCRIPTION
    )
    x_accounts: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_X_ACCOUNTS_DESCRIPTION
    )
    reddit_subreddits: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_REDDIT_SUBREDDITS_DESCRIPTION
    )
    youtube_channels: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_YOUTUBE_CHANNELS_DESCRIPTION
    )
    tiktok_count: Optional[Literal[10, 20, 30]] = Field(
        default=None, description=_MONITOR_TIKTOK_COUNT_DESCRIPTION
    )
    tiktok_accounts: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_TIKTOK_ACCOUNTS_DESCRIPTION
    )
    tiktok_hashtags: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_TIKTOK_HASHTAGS_DESCRIPTION
    )
    threads_accounts: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_THREADS_ACCOUNTS_DESCRIPTION
    )
    instagram_accounts: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_INSTAGRAM_ACCOUNTS_DESCRIPTION
    )
    instagram_hashtags: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_INSTAGRAM_HASHTAGS_DESCRIPTION
    )
    facebook_pages: Optional[List[str]] = Field(
        default=None, max_length=10, description=_MONITOR_FACEBOOK_PAGES_DESCRIPTION
    )
    meta_ads_count: Optional[Literal[10, 20, 30]] = Field(
        default=None, description=_MONITOR_META_ADS_COUNT_DESCRIPTION
    )


class MonitorCreateToolInput(_MonitorOptionsInput):
    query: str = Field(min_length=3, max_length=500, description=_MONITOR_QUERY_DESCRIPTION)
    name: Optional[str] = Field(
        default=None, max_length=120, description="A label for the console and webhooks."
    )
    criteria: Optional[List[str]] = Field(
        default=None, max_length=5, description=_MONITOR_CRITERIA_DESCRIPTION
    )
    sources: Optional[List[_MonitorSource]] = Field(
        default=None, min_length=1, description=_MONITOR_SOURCES_DESCRIPTION
    )
    interval: Optional[_MonitorInterval] = Field(
        default=None, description=_MONITOR_INTERVAL_DESCRIPTION
    )
    min_score: Optional[int] = Field(
        default=None, ge=0, le=100, description=_MONITOR_MIN_SCORE_DESCRIPTION
    )
    summary: Optional[bool] = Field(default=None, description=_MONITOR_SUMMARY_DESCRIPTION)
    webhook_url: Optional[str] = Field(
        default=None, max_length=2000, description=_MONITOR_WEBHOOK_URL_DESCRIPTION
    )
    webhook_events: Optional[List[_MonitorEvent]] = Field(
        default=None, min_length=1, description=_MONITOR_WEBHOOK_EVENTS_DESCRIPTION
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None, description=_MONITOR_METADATA_DESCRIPTION
    )
    run_now: bool = Field(
        default=True,
        description=(
            "Queue the first run at once (default true; that run is billed). False waits "
            "for the first scheduled run."
        ),
    )

    @model_validator(mode="after")
    def check_body(self) -> "MonitorCreateToolInput":
        # Refuse values the API would reject before anything is created.
        monitor_create_body(**self.model_dump())
        return self


class MonitorCreateTool(_MonitorTool):
    name: str = MONITOR_CREATE_TOOL
    description: str = (
        "Creates a Desearch monitor: a search that reruns on a schedule (every 1h to 7d) "
        "over the web, X, Reddit, YouTube, Threads, TikTok and Meta ads (meta_ads, the Meta "
        "Ad Library), or watches accounts, TikTok and Instagram hashtags and Facebook pages, "
        "and keeps only what is new. Each run searches the time since the previous run (or "
        "reads the newest posts of watched accounts and hashtags: x_accounts, "
        "reddit_subreddits, youtube_channels, tiktok_accounts, tiktok_hashtags, "
        "threads_accounts, instagram_accounts, instagram_hashtags, facebook_pages; a "
        "hashtag feed lists posts from any account), drops everything the monitor has seen "
        "before, screens the rest with an AI classifier for relevance to `query` and each of "
        "`criteria`, and keeps items scoring at least `min_score` (0 keeps every new post). "
        "The instagram source needs instagram_accounts or instagram_hashtags and the "
        "facebook source facebook_pages (HTTP 422 without). "
        f"{_MONITOR_META_ADS_NOTE} {_MONITOR_TIKTOK_SEARCH_NOTE} Kept items go "
        "to `webhook_url` (signed) and "
        f"stay readable with {MONITOR_ITEMS_TOOL}. {_MONITOR_BILLING} With run_now (the "
        "default) the first run is queued at once. Returns the monitor with "
        "max_run_cost_usd and, when a webhook is set, webhook_secret (shown only once). "
        f"Check the cost first with {MONITOR_ESTIMATE_TOOL}. At most 25 monitors per account."
    )
    args_schema: ArgsSchema = MonitorCreateToolInput

    def _run(
        self,
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
        run_now: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        body = monitor_create_body(
            query,
            name=name,
            criteria=criteria,
            sources=sources,
            interval=interval,
            min_score=min_score,
            summary=summary,
            web_count=web_count,
            x_count=x_count,
            x_lang=x_lang,
            x_min_likes=x_min_likes,
            x_accounts=x_accounts,
            reddit_subreddits=reddit_subreddits,
            youtube_channels=youtube_channels,
            tiktok_count=tiktok_count,
            tiktok_accounts=tiktok_accounts,
            tiktok_hashtags=tiktok_hashtags,
            threads_accounts=threads_accounts,
            instagram_accounts=instagram_accounts,
            instagram_hashtags=instagram_hashtags,
            facebook_pages=facebook_pages,
            meta_ads_count=meta_ads_count,
            webhook_url=webhook_url,
            webhook_events=webhook_events,
            metadata=metadata,
            run_now=run_now,
        )
        return self._call(create_monitor, body=body)

    async def _arun(
        self,
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
        run_now: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        body = monitor_create_body(
            query,
            name=name,
            criteria=criteria,
            sources=sources,
            interval=interval,
            min_score=min_score,
            summary=summary,
            web_count=web_count,
            x_count=x_count,
            x_lang=x_lang,
            x_min_likes=x_min_likes,
            x_accounts=x_accounts,
            reddit_subreddits=reddit_subreddits,
            youtube_channels=youtube_channels,
            tiktok_count=tiktok_count,
            tiktok_accounts=tiktok_accounts,
            tiktok_hashtags=tiktok_hashtags,
            threads_accounts=threads_accounts,
            instagram_accounts=instagram_accounts,
            instagram_hashtags=instagram_hashtags,
            facebook_pages=facebook_pages,
            meta_ads_count=meta_ads_count,
            webhook_url=webhook_url,
            webhook_events=webhook_events,
            metadata=metadata,
            run_now=run_now,
        )
        return await self._acall(create_monitor, body=body)


class MonitorEstimateToolInput(_MonitorOptionsInput):
    sources: Optional[List[_MonitorSource]] = Field(
        default=None, min_length=1, description=_MONITOR_SOURCES_DESCRIPTION
    )
    interval: Optional[_MonitorInterval] = Field(
        default=None, description=_MONITOR_INTERVAL_DESCRIPTION
    )
    summary: Optional[bool] = Field(default=None, description=_MONITOR_SUMMARY_DESCRIPTION)


class MonitorEstimateTool(_MonitorTool):
    name: str = MONITOR_ESTIMATE_TOOL
    description: str = (
        "Estimates what a Desearch monitor with these sources, options (watched accounts "
        "and hashtags included), summary and interval can cost: max_run_cost_usd (the most one run can cost, held on the "
        "balance while it runs), runs_per_month, max_monthly_cost_usd and a breakdown per "
        "part of a run (each source, screening and the summary). Runs usually cost less, "
        "since a run bills only what it used. It prices instagram and facebook without "
        "accounts or hashtags as one account each. Free and instant: it creates nothing."
    )
    args_schema: ArgsSchema = MonitorEstimateToolInput

    def _run(
        self,
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        body = monitor_estimate_body(
            sources,
            interval,
            summary,
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
        return self._call(estimate_monitor, body=body)

    async def _arun(
        self,
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
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        body = monitor_estimate_body(
            sources,
            interval,
            summary,
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
        return await self._acall(estimate_monitor, body=body)


class MonitorsToolInput(BaseModel):
    monitor_id: Optional[str] = Field(
        default=None,
        description="Read this one monitor instead of listing them.",
    )
    status: Optional[List[Literal["active", "paused"]]] = Field(
        default=None, description="Only monitors in these statuses: active, paused."
    )
    limit: int = Field(default=20, ge=1, le=100, description="Monitors per page (1-100).")
    cursor: Optional[str] = Field(
        default=None,
        description="next_cursor from the previous page. Keep the same status when you page.",
    )


class MonitorsTool(_MonitorTool):
    name: str = MONITORS_TOOL
    description: str = (
        "Lists your Desearch monitors, newest first, or reads one by monitor_id: query, "
        "criteria, sources, interval, status (active or paused, with paused_reason), "
        "webhook (url, events, consecutive_failures and last_delivery: the newest delivery "
        "with its status pending, delivered or failed, attempts, status_code, error and "
        "next_attempt_at; null before the first), max_run_cost_usd, next_run_at, the last "
        "run and totals (runs, items, spent_usd). Pages with next_cursor. Free and "
        "instant: it never starts a run."
    )
    args_schema: ArgsSchema = MonitorsToolInput

    def _run(
        self,
        monitor_id: Optional[str] = None,
        status: Optional[List[str]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return self._call(
            get_monitors, monitor_id=monitor_id, status=status, limit=limit, cursor=cursor
        )

    async def _arun(
        self,
        monitor_id: Optional[str] = None,
        status: Optional[List[str]] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await self._acall(
            get_monitors, monitor_id=monitor_id, status=status, limit=limit, cursor=cursor
        )


class MonitorUpdateToolInput(_MonitorOptionsInput):
    monitor_id: str = Field(min_length=1, description=_MONITOR_ID_DESCRIPTION)
    query: Optional[str] = Field(
        default=None, min_length=3, max_length=500, description=_MONITOR_QUERY_DESCRIPTION
    )
    name: Optional[str] = Field(
        default=None, max_length=120, description="A label for the console and webhooks."
    )
    criteria: Optional[List[str]] = Field(
        default=None,
        max_length=5,
        description=f"{_MONITOR_CRITERIA_DESCRIPTION} Replaces the current criteria; [] removes them.",
    )
    sources: Optional[List[_MonitorSource]] = Field(
        default=None, min_length=1, description=_MONITOR_SOURCES_DESCRIPTION
    )
    interval: Optional[_MonitorInterval] = Field(
        default=None, description=_MONITOR_INTERVAL_DESCRIPTION
    )
    min_score: Optional[int] = Field(
        default=None, ge=0, le=100, description=_MONITOR_MIN_SCORE_DESCRIPTION
    )
    summary: Optional[bool] = Field(default=None, description=_MONITOR_SUMMARY_DESCRIPTION)
    webhook_url: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            f"{_MONITOR_WEBHOOK_URL_DESCRIPTION} A new URL gets a new webhook_secret, "
            "returned once."
        ),
    )
    webhook_events: Optional[List[_MonitorEvent]] = Field(
        default=None, min_length=1, description=_MONITOR_WEBHOOK_EVENTS_DESCRIPTION
    )
    remove_webhook: bool = Field(
        default=False, description="Remove the monitor's webhook."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description=f"{_MONITOR_METADATA_DESCRIPTION} Replaces the current labels.",
    )

    @model_validator(mode="after")
    def check_changes(self) -> "MonitorUpdateToolInput":
        if self.remove_webhook and (self.webhook_url or self.webhook_events):
            raise ValueError("Set remove_webhook or webhook_url/webhook_events, not both.")
        changes = self.model_dump(exclude={"monitor_id", "remove_webhook"}, exclude_none=True)
        if not changes and not self.remove_webhook:
            raise ValueError("Nothing to change: set at least one setting.")
        # Refuse values the API would reject before anything is sent.
        _monitor_update_kwargs(**self.model_dump())
        return self


def _monitor_update_kwargs(
    monitor_id: str,
    query: Optional[str] = None,
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
    remove_webhook: bool = False,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if webhook_url:
        monitor_webhook(webhook_url, webhook_events)
    return {
        "monitor_id": monitor_id,
        "settings": monitor_settings(
            query, name, criteria, sources, interval, min_score, summary, metadata
        ),
        "options": monitor_options(
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
        ),
        "webhook_url": webhook_url,
        "webhook_events": webhook_events,
        "remove_webhook": remove_webhook,
    }


class MonitorUpdateTool(_MonitorTool):
    name: str = MONITOR_UPDATE_TOOL
    description: str = (
        "Changes a Desearch monitor's settings: query, name, criteria, sources, interval, "
        "min_score, summary, the per-source options, the webhook (a new URL gets a new "
        "webhook_secret, returned once; remove_webhook removes it) and metadata. Only the "
        "settings you set change; the per-source options and webhook events are merged with "
        "the current ones, and a new account or hashtag list replaces that source's watched "
        "accounts or hashtags ([] stops watching). The API answers HTTP 422 when the monitor "
        "as changed would have the instagram source without accounts or hashtags, or the "
        "facebook source without pages. Free. To pause, resume, "
        f"run now or delete a monitor, use {MONITOR_CONTROL_TOOL}."
    )
    args_schema: ArgsSchema = MonitorUpdateToolInput

    def _run(
        self,
        monitor_id: str,
        query: Optional[str] = None,
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
        remove_webhook: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        kwargs = _monitor_update_kwargs(
            monitor_id,
            query=query,
            name=name,
            criteria=criteria,
            sources=sources,
            interval=interval,
            min_score=min_score,
            summary=summary,
            web_count=web_count,
            x_count=x_count,
            x_lang=x_lang,
            x_min_likes=x_min_likes,
            x_accounts=x_accounts,
            reddit_subreddits=reddit_subreddits,
            youtube_channels=youtube_channels,
            tiktok_count=tiktok_count,
            tiktok_accounts=tiktok_accounts,
            tiktok_hashtags=tiktok_hashtags,
            threads_accounts=threads_accounts,
            instagram_accounts=instagram_accounts,
            instagram_hashtags=instagram_hashtags,
            facebook_pages=facebook_pages,
            meta_ads_count=meta_ads_count,
            webhook_url=webhook_url,
            webhook_events=webhook_events,
            remove_webhook=remove_webhook,
            metadata=metadata,
        )
        return self._call(update_monitor, **kwargs)

    async def _arun(
        self,
        monitor_id: str,
        query: Optional[str] = None,
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
        remove_webhook: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        kwargs = _monitor_update_kwargs(
            monitor_id,
            query=query,
            name=name,
            criteria=criteria,
            sources=sources,
            interval=interval,
            min_score=min_score,
            summary=summary,
            web_count=web_count,
            x_count=x_count,
            x_lang=x_lang,
            x_min_likes=x_min_likes,
            x_accounts=x_accounts,
            reddit_subreddits=reddit_subreddits,
            youtube_channels=youtube_channels,
            tiktok_count=tiktok_count,
            tiktok_accounts=tiktok_accounts,
            tiktok_hashtags=tiktok_hashtags,
            threads_accounts=threads_accounts,
            instagram_accounts=instagram_accounts,
            instagram_hashtags=instagram_hashtags,
            facebook_pages=facebook_pages,
            meta_ads_count=meta_ads_count,
            webhook_url=webhook_url,
            webhook_events=webhook_events,
            remove_webhook=remove_webhook,
            metadata=metadata,
        )
        return await self._acall(update_monitor, **kwargs)


class MonitorControlToolInput(BaseModel):
    monitor_id: str = Field(min_length=1, description=_MONITOR_ID_DESCRIPTION)
    action: Literal[
        "pause", "resume", "trigger", "delete", "test_webhook", "rotate_webhook_secret"
    ] = Field(
        description=(
            "'pause' (no scheduled runs until resumed), 'resume' (also after a pause for "
            "credits or a failing webhook), 'trigger' (queue one billed run now), 'delete' "
            "(cannot be undone), 'test_webhook' (send a signed test event and report what "
            "the endpoint answered) or 'rotate_webhook_secret' (a new signing secret, "
            "returned once)."
        )
    )


class MonitorControlTool(_MonitorTool):
    name: str = MONITOR_CONTROL_TOOL
    description: str = (
        "Pauses, resumes, triggers or deletes a Desearch monitor, or tests or rotates its "
        "webhook. Pause, resume, delete, test_webhook and rotate_webhook_secret are free. "
        "trigger queues one run now and is BILLED like a scheduled run (up to the "
        "monitor's max_run_cost_usd); it fails with HTTP 409 while a run is queued or "
        f"running. Wait for a triggered run with {MONITOR_RUNS_TOOL} and its run_id. Delete "
        "cannot be undone."
    )
    args_schema: ArgsSchema = MonitorControlToolInput

    def _run(
        self,
        monitor_id: str,
        action: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return self._call(control_monitor, monitor_id=monitor_id, action=action)

    async def _arun(
        self,
        monitor_id: str,
        action: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await self._acall(control_monitor, monitor_id=monitor_id, action=action)


class MonitorRunsToolInput(BaseModel):
    monitor_id: str = Field(min_length=1, description=_MONITOR_ID_DESCRIPTION)
    run_id: Optional[str] = Field(
        default=None,
        description="Read this one run with the items it kept instead of listing runs.",
    )
    timeout: int = Field(
        default=0,
        ge=0,
        le=600,
        description=(
            "With run_id: seconds to wait for a queued or running run to end (default 0, "
            "no wait). Waiting is free."
        ),
    )
    limit: int = Field(default=20, ge=1, le=100, description="Runs per page (1-100).")
    cursor: Optional[str] = Field(
        default=None, description="next_cursor from the previous page of runs."
    )


class MonitorRunsTool(_MonitorTool):
    name: str = MONITOR_RUNS_TOOL
    description: str = (
        "Lists a Desearch monitor's runs, newest first, or reads one run by run_id with "
        "the items it kept (most relevant first). Each run has its status (queued, running, "
        "completed, failed, cancelled) and fail_reason, the time window it searched, per "
        "source how many items were fetched, new and kept (and accounts_failed and "
        "hashtags_failed: watched accounts and hashtags that could not be read), how items "
        "were screened (stats.screen: ai, by an AI classifier, or keyword when it fell "
        "back), item_count, the summary (when "
        "on), and reserved_usd and cost_usd. With run_id and timeout it waits for the run "
        "to end. Free: it never starts a run."
    )
    args_schema: ArgsSchema = MonitorRunsToolInput
    poll_interval: float = 5.0

    def _run(
        self,
        monitor_id: str,
        run_id: Optional[str] = None,
        timeout: int = 0,
        limit: int = 20,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return self._call(
            read_runs,
            monitor_id=monitor_id,
            run_id=run_id,
            limit=limit,
            cursor=cursor,
            timeout=timeout,
            poll_interval=self.poll_interval,
        )

    async def _arun(
        self,
        monitor_id: str,
        run_id: Optional[str] = None,
        timeout: int = 0,
        limit: int = 20,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await self._acall(
            read_runs,
            monitor_id=monitor_id,
            run_id=run_id,
            limit=limit,
            cursor=cursor,
            timeout=timeout,
            poll_interval=self.poll_interval,
        )


class MonitorItemsToolInput(BaseModel):
    monitor_id: str = Field(min_length=1, description=_MONITOR_ID_DESCRIPTION)
    source: Optional[_MonitorSource] = Field(
        default=None,
        description=(
            "Only items from this source: web, x, reddit, youtube, threads, tiktok, "
            "instagram, facebook, meta_ads."
        ),
    )
    limit: int = Field(default=50, ge=1, le=200, description="Items per page (1-200).")
    cursor: Optional[str] = Field(
        default=None,
        description="next_cursor from the previous page. Keep the same source when you page.",
    )


class MonitorItemsTool(_MonitorTool):
    name: str = MONITOR_ITEMS_TOOL
    description: str = (
        "Reads everything a Desearch monitor has kept, newest first: per item its source, "
        "url, title, text (cut to 500 characters), author, published_at, score (0-100) "
        "with the screening signals (relevant, new_info, and criteria when the monitor has "
        "criteria), and source data (X likes, reposts, replies, views; Reddit subreddit, "
        "score, comments; YouTube channel, views, thumbnail; TikTok views, likes, comments, "
        "shares, saves, duration_s; Threads likes; Instagram likes, comments, views, "
        "media_type; Facebook reactions, comments, shares, views; TikTok, Instagram and "
        "Facebook post_id (the ID the API's GET /desearch/social/post gives the same post; "
        "an Instagram shortcode), type (video, image, carousel, reel, ...), hashtags, "
        "mentions (TikTok and Instagram), media (links to the post's videos and pictures) "
        "and thumbnail; Meta ads page, active, cta, link (the landing page), platforms, "
        "thumbnail, with the ad headline as title and url the Ad Library record; web domain, "
        "highlights; account, for an item read from a watched account, and hashtag, for one "
        "read from a watched hashtag). Media and thumbnail links expire after a few days; "
        "metrics are the post's when the run found it. "
        "Filter by source; page with next_cursor. Free: it never starts a run."
    )
    args_schema: ArgsSchema = MonitorItemsToolInput

    def _run(
        self,
        monitor_id: str,
        source: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool synchronously."""
        del run_manager
        return self._call(
            read_items, monitor_id=monitor_id, source=source, limit=limit, cursor=cursor
        )

    async def _arun(
        self,
        monitor_id: str,
        source: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        del run_manager
        return await self._acall(
            read_items, monitor_id=monitor_id, source=source, limit=limit, cursor=cursor
        )
