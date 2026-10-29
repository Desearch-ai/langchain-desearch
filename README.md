# LangChain Desearch Integration

This project integrates the Desearch API with LangChain tools to enable various search and data-fetching functionalities, such as web searches, Twitter data retrieval, and AI-powered searches.

## Features

- **Grouped Tools**:
  - **Search Tools**: General-purpose search tools: AI search, web and X search, AI link search, the Search API and page extraction.
  - **Twitter Tools**: Tools specifically for Twitter-related operations, including trends and searching one user's posts.
  - **Social Tools**: Public TikTok, Facebook and Instagram data, one tool per platform, with cursor paging.
  - **People Tools**: Find people on X, LinkedIn, YouTube and TikTok who match a plain-language description, and browse everyone your searches found.

## Installation

Install the package using pip:

```bash
pip install langchain-desearch
```

> **Release 1.2.0 waits for desearch-py 1.8.0 on PyPI.** langchain-desearch 1.2.0 requires `desearch-py>=1.8.0,<2.0.0`, the first release with the TikTok, Facebook, Instagram and Search API methods. desearch-py 1.8.0 is not on PyPI yet, so do not publish langchain-desearch 1.2.0 until it is. Until then, install desearch-py 1.8.0 from a checkout of the [desearch.py](https://github.com/Desearch-ai/desearch.py) repository (`pip install -e /path/to/desearch.py`) before installing this package from source. desearch-py 1.8.0 needs Python 3.10 or newer, and so does this release.

## Usage

### Tools

Every tool is importable from `langchain_desearch.tools`, reads `DESEARCH_API_KEY` from the environment, and works with `invoke` and `ainvoke`. The tools added in 1.2.0 are also exported from `langchain_desearch`.

| Tool class | Tool name | What it returns | desearch-py method | Group |
| --- | --- | --- | --- | --- |
| `DesearchTool` | `desearch_tool` | AI search over web and X with a written summary | `ai_search` | `search_tools` |
| `BasicWebSearchTool` | `basic_web_search_tool` | Web search results (title, link, snippet) | `web_search` | `search_tools` |
| `BasicTwitterSearchTool` | `basic_twitter_search_tool` | X posts matching a query and filters | `x_search` | `search_tools`, `twitter_tools` |
| `AIWebLinksSearchTool` | `ai_web_links_search_tool` | AI-picked web links (`search_results`), no summary | `ai_web_links_search` | `search_tools` |
| `AIXLinksSearchTool` | `ai_x_links_search_tool` | AI-picked X posts (`miner_tweets`), no summary | `ai_x_links_search` | `search_tools`, `twitter_tools` |
| `SearchApiTool` | `search_api_tool` | Ranked pages, each with the passage that best matches the query (`POST /search`) | `search_api` | `search_tools` |
| `WebExtractTool` | `web_extract_tool` | A page's text or HTML (`GET /web/extract`) | `extract` | `search_tools` |
| `FetchTweetsByUrlsTool` | `fetch_tweets_by_urls_tool` | Posts by URL | `x_posts_by_urls` | `twitter_tools` |
| `FetchTweetsByIdTool` | `fetch_tweets_by_id_tool` | One post by ID | `x_post_by_id` | `twitter_tools` |
| `FetchLatestTweetsTool` | `fetch_latest_tweets_tool` | A user's latest posts | `x_user_posts` | `twitter_tools` |
| `FetchTweetsAndRepliesByUserTool` | `fetch_tweets_and_replies_by_user_tool` | A user's posts and replies | `x_user_replies` | `twitter_tools` |
| `FetchRepliesByPostTool` | `fetch_replies_by_post_tool` | Replies to a post | `x_post_replies` | `twitter_tools` |
| `FetchRetweetsByPostTool` | `fetch_retweets_by_post_tool` | Users who reposted a post | `x_post_retweeters` | `twitter_tools` |
| `FetchTwitterUserTool` | `fetch_twitter_user_tool` | A user's profile and timeline | `x_user_posts` | `twitter_tools` |
| `SearchTweetsByUserTool` | `search_tweets_by_user_tool` | One user's posts matching a query (`GET /twitter/post/user`) | `x_posts_by_user` | `twitter_tools` |
| `FetchTwitterTrendsTool` | `fetch_twitter_trends_tool` | Trending topics for a WOEID | `x_trends` | `twitter_tools` |
| `TikTokTool` | `tiktok_tool` | TikTok posts, profiles, comments, hashtags, trending | `tiktok_*` (7 actions) | `social_tools` |
| `FacebookTool` | `facebook_tool` | Facebook posts, profiles, comments; Ad Library search | `facebook_*` (6 actions) | `social_tools` |
| `InstagramTool` | `instagram_tool` | Instagram posts, profiles, comments, hashtags; profile search | `instagram_*` (6 actions) | `social_tools` |
| `PeopleSearchTool` | `people_search_tool` | People matching a description (billed, 1-5 minutes) | HTTP `/desearch/people` | `people_tools` |
| `PeopleProfilesTool` | `people_profiles_tool` | People earlier searches found (free) | HTTP `/desearch/people` | `people_tools` |

All tools set `return_direct=True`, so an agent returns their output as its answer. Create a tool with `return_direct=False` (for example `TikTokTool(return_direct=False)`) if the agent should read the result and keep going.

### Grouped Tools

The groups are lists in `langchain_desearch.search_tools`.

#### Search Tools
The `search_tools` group contains tools for general-purpose searches. `create_search_agent` uses this list.
- `DesearchTool`: Perform AI searches over web and X with a written summary.
- `BasicWebSearchTool`: Conduct basic web searches.
- `BasicTwitterSearchTool`: Perform advanced Twitter searches with filters.
- `AIWebLinksSearchTool`: Find web links for a prompt with Desearch AI, without a summary. Put `site:<domain>` in the prompt to keep results to one site.
- `AIXLinksSearchTool`: Find X posts for a prompt with Desearch AI, without a summary.
- `SearchApiTool`: Search the web and get ranked pages, each with its best matching passage. Takes `mode` (`fast` or `balanced`), `count` (10, 15 or 20), `start_date`/`end_date` (`YYYY-MM-DD`), `highlights` and `page_text`.
- `WebExtractTool`: Read a public page as text (default) or HTML. Set `js=True` (and optionally `wait` milliseconds) for pages that need JavaScript.

#### Twitter Tools
The `twitter_tools` group contains tools specifically for Twitter-related operations:
- `BasicTwitterSearchTool`: Perform a basic Twitter search using Desearch.
- `FetchTweetsByUrlsTool`: Retrieve tweets from specific URLs.
- `FetchTweetsByIdTool`: Fetch tweets using their unique IDs.
- `FetchLatestTweetsTool`: Get the latest tweets from a specific user.
- `FetchTweetsAndRepliesByUserTool`: Retrieve tweets and replies from a user.
- `FetchRepliesByPostTool`: Fetch replies to a specific Twitter post.
- `FetchRetweetsByPostTool`: Retrieve users who retweeted a specific post.
- `FetchTwitterUserTool`: Get detailed information about a Twitter user.
- `SearchTweetsByUserTool`: Search one user's posts, optionally matching a query.
- `FetchTwitterTrendsTool`: Get trending topics for a location by WOEID (Worldwide is 1, the default; the United States is 23424977).
- `AIXLinksSearchTool`: Find X posts for a prompt with Desearch AI.

#### Social Tools
The `social_tools` group has one tool per platform: `TikTokTool` (`tiktok_tool`), `FacebookTool` (`facebook_tool`) and `InstagramTool` (`instagram_tool`). Each takes an `action` and the fields that action needs, so an agent gets three tools instead of nineteen.

| Tool | `action` | Required | Optional | desearch-py method | Pages |
| --- | --- | --- | --- | --- | --- |
| `tiktok_tool` | `search` | `query` | `count`, `cursor`, `sort` | `tiktok_search` | yes |
| | `post` | `id` (video ID) | | `tiktok_post_by_id` | |
| | `profile` | `username` | | `tiktok_profile` | |
| | `profile_posts` | `username` | `count`, `cursor` | `tiktok_profile_posts` | yes |
| | `comments` | `video_id` (a post's `id`) | `count`, `cursor` | `tiktok_comments` | yes |
| | `hashtag` | `tag` | `count`, `cursor` | `tiktok_hashtag` | yes |
| | `trending` | | `region` (default `US`), `count` | `tiktok_trending` | no |
| `facebook_tool` | `search` | `query` | `count`, `cursor` | `facebook_search` (Meta Ad Library) | yes |
| | `post` | `id` | | `facebook_post_by_id` | |
| | `profile` | `profile_id` (ID, username or handle) | | `facebook_profile` | |
| | `profile_posts` | `profile_id` | `count`, `cursor` | `facebook_profile_posts` | yes |
| | `comments` | `post_id` | `count`, `cursor` | `facebook_comments` | yes |
| | `hashtag` | `tag` | `count`, `cursor` | `facebook_hashtag` (Meta Ad Library) | yes |
| `instagram_tool` | `search` | `query` | `count` | `instagram_search` (profiles) | no |
| | `post` | `id` (shortcode, media ID or post URL) | | `instagram_media` | |
| | `profile` | `username` | | `instagram_profile` | |
| | `profile_posts` | `username` | `count`, `cursor` | `instagram_profile_posts` | yes |
| | `comments` | `shortcode` (the code in `instagram.com/p/<shortcode>/`) | `count`, `cursor` | `instagram_comments` | yes |
| | `hashtag` | `tag` | `count`, `cursor` | `instagram_hashtag` | yes |

- `post` and `profile` return one object. The other actions return `{"results": [...], "next_cursor": ...}`.
- `count` is a maximum (1-100). The API default is 10 for `search` and 20 elsewhere. A page can hold fewer items without being the last one.
- To read the next page, call the tool again with the same action and target and `cursor` set to `next_cursor`. `next_cursor` is `None` on the last page, and on `tiktok_tool` `trending` and `instagram_tool` `search`, which do not paginate. A cursor is opaque and is sent back unchanged; it is valid for 24 hours.
- A missing target, a field the action does not use, or a `cursor` on an action that does not page raises a validation error that names the fields the action takes, before any request is sent. Empty strings count as unset.
- A leading `@` is removed from `username` and `profile_id`, and a leading `#` from `tag`.
- `facebook_tool` `search` and `hashtag` read the public Meta Ad Library, not organic posts.

#### People Tools
The `people_tools` group contains the [People Search](https://www.desearch.ai/docs/guide/apis/people-search) tools:
- `PeopleSearchTool` (`people_search_tool`): Start a People Search on `x`, `linkedin`, `youtube` or `tiktok` and wait for it. Returns compact people: name, handle, url, platform, audience, location, fit score, why they matched, contacts and links.
- `PeopleProfilesTool` (`people_profiles_tool`): List the people your earlier searches found on one platform, with filters (`has=["email"]`, audience range, minimum score, location, verified) and sorting. Free.

> **Billing and timing**: `people_search_tool` is billed. Starting a search reserves `count` profiles (5-200) at the route price; you pay only for the profiles it delivers, and a search that finds no one is free. A search runs in the background for 1-5 minutes. The tool waits up to `timeout` seconds (default 300), then returns the people found so far with the `task_id`. Call it again with only `task_id` to keep waiting: that is free and does not start a new search. `people_tools` is kept out of `search_tools`, so `create_search_agent` never starts a billed search.

### Examples

#### Using Tools
```python
from langchain_desearch.tools import DesearchTool, BasicWebSearchTool, BasicTwitterSearchTool
from dotenv import load_dotenv
load_dotenv()

# Example 1: Using DesearchTool
tool = DesearchTool()
result = tool._run(
    prompt="Bittensor",
    tool=['web'],
    model="NOVA",
    date_filter="PAST_24_HOURS",
    streaming=False
)
print(result)

# Example 2: Using BasicWebSearchTool
tool = BasicWebSearchTool()
result = tool._run(
    query="Latest news on AI",
    num=5,
    start=1
)
print(result)

# Example 3: Using BasicTwitterSearchTool
tool = BasicTwitterSearchTool()
result = tool._run(
    query="AI trends",
    sort="Top",
    count=5
)
print(result)
```

#### Using the Web, AI Links and X Tools
```python
from langchain_desearch import (
    AIWebLinksSearchTool,
    AIXLinksSearchTool,
    FetchTwitterTrendsTool,
    SearchApiTool,
    SearchTweetsByUserTool,
    WebExtractTool,
)
from dotenv import load_dotenv
load_dotenv()

# Ranked pages, each with the passage that best matches the query.
search = SearchApiTool().invoke({
    "query": "Who won the 2024 Nobel Prize in Physics?",
    "mode": "fast",
    "count": 10,
    "start_date": "2024-01-01",
})
for result in search["results"]:
    print(result["title"], result["url"], result.get("highlights"))

# Read one page as plain text.
text = WebExtractTool().invoke({"url": "https://desearch.ai", "format": "text"})

# Links only, picked by Desearch AI.
links = AIWebLinksSearchTool().invoke({"prompt": "Bittensor subnet 22", "count": 10})
for item in links["search_results"]:
    print(item["title"], item["link"])
posts = AIXLinksSearchTool().invoke({"prompt": "Bittensor news", "count": 10})
print(len(posts["miner_tweets"]))

# One user's posts matching a query, and the trends in the United States.
tweets = SearchTweetsByUserTool().invoke({"user": "elonmusk", "query": "tesla", "count": 5})
trends = FetchTwitterTrendsTool().invoke({"woeid": 23424977})
print([trend["name"] for trend in trends["trends"]][:10])
```

#### Using the Social Tools
```python
from langchain_desearch import FacebookTool, InstagramTool, TikTokTool
from dotenv import load_dotenv
load_dotenv()

tiktok = TikTokTool()
profile = tiktok.invoke({"action": "profile", "username": "tiktok"})
print(profile["nickname"], profile["followers"])

# Read up to three pages of a hashtag, following next_cursor.
cursor = None
for _ in range(3):
    page = tiktok.invoke({"action": "hashtag", "tag": "ai", "count": 20, "cursor": cursor})
    for post in page["results"]:
        print(post["id"], post.get("text"))
    cursor = page["next_cursor"]
    if cursor is None:  # last page
        break

# A post's id is the video_id for its comments.
video_id = page["results"][0]["id"]
comments = tiktok.invoke({"action": "comments", "video_id": video_id, "count": 10})

# Instagram comments take the post shortcode.
instagram = InstagramTool()
posts = instagram.invoke({"action": "profile_posts", "username": "instagram", "count": 12})
shortcode = posts["results"][0]["shortcode"]
comments = instagram.invoke({"action": "comments", "shortcode": shortcode})

# Facebook search reads the Meta Ad Library.
ads = FacebookTool().invoke({"action": "search", "query": "running shoes", "count": 10})
print(len(ads["results"]), ads["next_cursor"])
```

To give an agent the social tools, pass `social_tools` from `langchain_desearch.search_tools`.

#### Using People Search
```python
from langchain_desearch.tools import PeopleSearchTool, PeopleProfilesTool
from dotenv import load_dotenv
load_dotenv()

# Start a billed search and wait up to 5 minutes for it.
people = PeopleSearchTool().invoke({
    "query": "Indie hackers building in public who share their MRR",
    "platform": "x",
    "count": 25,
    "min_score": 60,
    "min_followers": 1000,
    "timeout": 300,
})
print(people["status"], len(people["people"]))
for person in people["people"]:
    print(person["handle"], person.get("audience"), person.get("why"))

# Still running? Keep waiting on the same search (free, nothing new is billed).
if not people["done"]:
    people = PeopleSearchTool().invoke({"task_id": people["task_id"], "timeout": 300})

# Browse everyone your X searches found who has a public email (free).
library = PeopleProfilesTool().invoke({
    "platform": "x",
    "has": ["email"],
    "min_audience": 1000,
    "sort": "score",
    "limit": 25,
})
print(library["total"], library.get("next_cursor"))
```

Platform filters for `people_search_tool`:

| Platform | Filters |
| --- | --- |
| `x` | `min_followers`, `max_followers`, `verified_only`, `countries`, `keywords`, `exclude_keywords` |
| `linkedin` | `min_followers`, `max_followers`, `countries`, `keywords`, `exclude_keywords` |
| `youtube` | `min_subscribers`, `max_subscribers`, `countries`, `require_contact`, `keywords`, `exclude_keywords` |
| `tiktok` | `min_followers`, `max_followers`, `require_contact`, `keywords`, `exclude_keywords` |

A filter that does not apply to the platform is rejected before any request is sent, so nothing is billed. Every search also takes `count`, `min_score`, `exclude` (handles to leave out) and `idempotency_key` (sending the same key again returns the first search instead of billing a new one).

To give an agent the People tools, pass `people_tools` from `langchain_desearch.search_tools` to your agent. Like the other tools, they set `return_direct=True`; create them with `PeopleSearchTool(return_direct=False)` if the agent should read the result and keep going.

> People Search calls the `/desearch/people` routes over HTTP with aiohttp, because the People Search methods in `desearch-py` arrive in 1.4.0, which is not on PyPI yet. The tools will switch to the SDK once it is published.

#### Using RAG (Retrieval-Augmented Generation)
```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_desearch.tools import DesearchTool
from langchain_deepseek import ChatDeepSeek

# Setup Desearch Tool
desearch_tool = DesearchTool()

# Template to wrap Desearch output
document_prompt = PromptTemplate.from_template("""
<source>
    <result>{result}</result>
</source>
""")

# Retrieval chain using DesearchTool
def get_desearch_context(prompt: str) -> str:
    return desearch_tool._run(prompt=prompt, tool=["web"], model="NOVA")

retrieval_chain = RunnableLambda(lambda query: {
    "result": get_desearch_context(query)
}) | document_prompt | (lambda docs: docs.text)

# Prompt for RAG generation
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research assistant. You use xml-formatted context to research people's questions."),
    ("human", """
Please answer the following query based on the provided context. Please cite your sources at the end of your response.:

Query: {query}
---
<context>
{context}
</context>
""")
])

# Use DeepSeek for LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

output_parser = StrOutputParser()

# Final chain
chain = RunnableParallel({
    "query": RunnablePassthrough(),
    "context": retrieval_chain,
}) | generation_prompt | llm | output_parser

# Run it!
query = "Recent trends in AI safety research"
result = chain.invoke(query)
print(result)
```

#### Using the LangChain Agent
```python
from langchain_desearch.agent import create_search_agent
from langchain_deepseek import ChatDeepSeek

# Create a DeepSeek LLM instance
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Initialize the search agent
search_agent = create_search_agent(llm=llm)

# Use the agent to perform a task
state = {
    "input_message": "What's the latest news on AI?",
}
response = search_agent.invoke(state)
print(f"Agent Response: {response['output']}")
```

### Running Tests

#### Dummy Tests
Run the dummy tests to verify the tools' functionality with mocked data:
```bash
pytest tests/unit_tests
```

#### Real API Tests
Run the real tests to verify the tools' functionality with the Desearch API:
```bash
pytest tests/integration_tests
```

> **Note**: Ensure you have a valid `DESEARCH_API_KEY` in your `.env` file before running real tests.

#### End-to-end smoke test
`scripts/e2e-full-coverage.py` calls every tool added in 1.2.0 once against the live API through `invoke` (one call through `ainvoke`), with the smallest counts, follows one social cursor page, and prints `PASS <name>` or `FAIL <name>: <error>` per call. It exits non-zero if any call fails. It never starts a People Search (a billed job) unless you pass `--with-people`.
```bash
DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py
DESEARCH_API_KEY=... python scripts/e2e-full-coverage.py --with-people  # also starts one billed People Search (count 5)
```

## Project Structure

```
langchain_desearch/
├── langchain_desearch/
│   ├── __init__.py
│   ├── tools.py
│   ├── people.py
│   ├── social.py
│   ├── search_tools.py
│   ├── agent.py
├── examples/
│   ├── tools.py
│   ├── RAG.py
│   ├── agent.py
├── scripts/
│   ├── e2e-full-coverage.py
├── tests/
│   ├── test_tools.py
│   ├── test_tools_real.py
├── .env
├── README.md
├── setup.py
├── requirements.txt
```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or support, please contact [your-email@example.com].