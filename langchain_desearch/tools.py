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
