# LangChain Desearch Integration

This project integrates the Desearch API with LangChain tools to enable various search and data-fetching functionalities, such as web searches, Twitter data retrieval, and AI-powered searches.

## Features

- **Grouped Tools**:
  - **Search Tools**: General-purpose search tools: AI search, web and X search, AI link search, the Search API and page extraction.
  - **Twitter Tools**: Tools specifically for Twitter-related operations, including trends and searching one user's posts.
  - **Social Tools**: Public TikTok, Facebook, Instagram, YouTube, Reddit and Threads data, one tool per platform, with cursor paging.
  - **People Tools**: Find people on X, LinkedIn, YouTube and TikTok who match a plain-language description, and browse everyone your searches found.
  - **Company Tools**: Find companies that match a plain-language description, or enrich the domains you have, with public emails, phones, social profiles and leaders.
  - **Monitor Tools**: Searches that rerun on a schedule over the web, X, Reddit, YouTube, Threads, TikTok and the Meta Ad Library (or watch accounts there, TikTok and Instagram hashtags, and accounts on Instagram and Facebook) and keep only what is new, with an optional signed webhook.

## Installation

Install the package using pip:

```bash
pip install langchain-desearch
```

> **Release 1.3.0 waits for desearch-py 1.11.0 on PyPI.** langchain-desearch 1.3.0 requires `desearch-py>=1.11.0,<2.0.0`, the first release with the YouTube, Reddit and Threads methods (it also carries the TikTok, Facebook, Instagram and Search API methods of desearch-py 1.8.0). desearch-py 1.11.0 is not on PyPI yet, so do not publish langchain-desearch 1.3.0 until it is. Until then, install desearch-py 1.11.0 from a checkout of the [desearch.py](https://github.com/Desearch-ai/desearch.py) repository (`pip install -e /path/to/desearch.py`) before installing this package from source. desearch-py 1.11.0 needs Python 3.10 or newer, and so does this release.

## Usage

### Tools

Every tool is importable from `langchain_desearch.tools`, reads `DESEARCH_API_KEY` from the environment, and works with `invoke` and `ainvoke`. The tools added in 1.2.0, 1.3.0 and 1.5.0 are also exported from `langchain_desearch`.

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
| `YouTubeTool` | `youtube_tool` | YouTube search, videos, transcripts, comments, channels and their uploads | `youtube_*` (6 actions) | `social_tools` |
| `RedditTool` | `reddit_tool` | Reddit search, subreddits and their posts, posts, comment trees | `reddit_*` (5 actions) | `social_tools` |
| `ThreadsTool` | `threads_tool` | Threads post and account search, profiles, latest posts, a post with replies | `threads_*` (5 actions) | `social_tools` |
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
The `social_tools` group has one tool per platform: `TikTokTool` (`tiktok_tool`), `FacebookTool` (`facebook_tool`), `InstagramTool` (`instagram_tool`), `YouTubeTool` (`youtube_tool`), `RedditTool` (`reddit_tool`) and `ThreadsTool` (`threads_tool`). Each takes an `action` and the fields that action needs, so an agent gets six tools instead of thirty-five.

| Tool | `action` | Required | Optional | desearch-py method | Pages |
| --- | --- | --- | --- | --- | --- |
| `tiktok_tool` | `search` | `query` | `count`, `cursor`, `sort` | `tiktok_search` | yes |
| | `post` | `id` (video ID) | | `tiktok_post_by_id` | |
| | `profile` | `username` | | `tiktok_profile` | |
| | `profile_posts` | `username` | `count`, `cursor` | `tiktok_profile_posts` | yes |
| | `comments` | `video_id` (a post's `id`) | `count`, `cursor` | `tiktok_comments` | yes |
| | `hashtag` | `tag` | `count`, `cursor` | `tiktok_hashtag` | yes |
| | `trending` | | `region` (default `US`), `count` | `tiktok_trending` | no |
| `facebook_tool` | `search` | `query` | `sort`, `count`, `cursor` | `facebook_search` (Meta Ad Library) | yes |
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
| `youtube_tool` | `search` | `query` | `type`, `upload_date`, `sort_by`, `duration`, `cursor` | `youtube_search` | yes |
| | `video` | `id` (video ID or URL) | | `youtube_video` | |
| | `transcript` | `video_id` | `language` | `youtube_transcript` | |
| | `comments` | `video_id` | `order`, `cursor` | `youtube_comments` | yes |
| | `channel` | `handle` (with or without `@`, or `UC…` ID) | | `youtube_channel` | |
| | `channel_videos` | `handle` | `sort`, `cursor` | `youtube_channel_videos` | yes |
| `reddit_tool` | `search` | `query` | `sort`, `timeframe`, `cursor` | `reddit_search` | yes |
| | `subreddit` | `name` (case-sensitive) | | `reddit_subreddit` | |
| | `subreddit_posts` | `name` | `sort`, `timeframe`, `cursor` | `reddit_subreddit_posts` | yes |
| | `post` | `id` (post ID or URL) | | `reddit_post_by_id` | |
| | `comments` | `post_id` | `cursor` | `reddit_comments` | yes |
| `threads_tool` | `search` | `query` | `start_date`, `end_date` | `threads_search` | no |
| | `search_users` | `query` | | `threads_search_users` | no |
| | `profile` | `username` | | `threads_profile` | |
| | `profile_posts` | `username` | | `threads_profile_posts` | no |
| | `post` | `id` (URL or shortcode) | | `threads_post_by_id` | |

- `post`, `profile`, `video`, `transcript`, `channel` and `subreddit` return one object. The other actions return `{"results": [...], "next_cursor": ...}`.
- `count` is a maximum (1-100). The API default is 10 for `search` and 20 elsewhere. A page can hold fewer items without being the last one. YouTube, Reddit and Threads take no `count`: a page is whatever the platform returns.
- To read the next page, call the tool again with the same action and target and `cursor` set to `next_cursor`. `next_cursor` is `None` on the last page, and on `tiktok_tool` `trending`, `instagram_tool` `search` and every `threads_tool` action, which do not paginate. A cursor is opaque and is sent back unchanged; it is valid for 24 hours.
- A missing target, a field the action does not use, or a `cursor` on an action that does not page raises a validation error that names the fields the action takes, before any request is sent. Empty strings count as unset.
- A leading `@` is removed from `username` and `profile_id`, a leading `#` from `tag`, and a leading `r/` from a subreddit `name`. A YouTube `handle` is sent as is.
- Fields with a fixed set of values (`type`, `upload_date`, `sort_by`, `duration`, `order`, `sort`, `timeframe`) are checked per action before any request, and `start_date`/`end_date` must be `YYYY-MM-DD`.
- `youtube_tool` `transcript` returns a not-found error (not charged) when the video has no captions. Each YouTube, Reddit and Threads call is billed as one request, including an empty page.
- `facebook_tool` `search` and `hashtag` read the public Meta Ad Library, not organic posts. `search` takes `sort`: `impressions` (the API default, most seen ads first) or `recent` (recently started ads first). `sort` needs desearch-py 1.13.0 or later; with an older desearch-py the tool returns an error saying so.

#### People Tools
The `people_tools` group contains the [People Search](https://www.desearch.ai/docs/guide/apis/people-search) tools:
- `PeopleSearchTool` (`people_search_tool`): Start a People Search on `x`, `linkedin`, `youtube` or `tiktok` and wait for it. Returns compact people: name, handle, url, platform, audience, location, fit score, why they matched, contacts and links.
- `PeopleProfilesTool` (`people_profiles_tool`): List the people your earlier searches found on one platform, with filters (`has=["email"]`, audience range, minimum score, location, verified) and sorting. Free.

> **Billing and timing**: `people_search_tool` is billed. Starting a search reserves `count` profiles (5-200) at the route price; you pay only for the profiles it delivers, and a search that finds no one is free. A search runs in the background for 1-5 minutes. The tool waits up to `timeout` seconds (default 300), then returns the people found so far with the `task_id`. Call it again with only `task_id` to keep waiting: that is free and does not start a new search. `people_tools` is kept out of `search_tools`, so `create_search_agent` never starts a billed search.

#### People Tools
The `people_tools` group contains the [People Search](https://www.desearch.ai/docs/guide/apis/people-search) tools:
- `PeopleSearchTool` (`people_search_tool`): Start a People Search on `x`, `linkedin`, `youtube` or `tiktok` and wait for it. Returns compact people: name, handle, url, platform, audience, location, fit score, why they matched, contacts and links.
- `PeopleProfilesTool` (`people_profiles_tool`): List the people your earlier searches found on one platform, with filters (`has=["email"]`, audience range, minimum score, location, verified) and sorting. Free.

> **Billing and timing**: `people_search_tool` is billed. Starting a search reserves `count` profiles (5-200) at the route price; you pay only for the profiles it delivers, and a search that finds no one is free. A search runs in the background for 1-5 minutes. The tool waits up to `timeout` seconds (default 300), then returns the people found so far with the `task_id`. Call it again with only `task_id` to keep waiting: that is free and does not start a new search. `people_tools` is kept out of `search_tools`, so `create_search_agent` never starts a billed search.

#### Company Tools
The `company_tools` group contains the [Company Search](https://www.desearch.ai/docs/guide/apis/company-search) tools:
- `CompanySearchTool` (`company_search_tool`): Start a Company Search from a plain-language description and wait for it. Returns compact companies: domain, name, url, location, fit score and why it matched, primary email and contact, public emails, phones and social profiles (each with the page it came from), and leaders with their LinkedIn profiles and emails.
- `CompanyEnrichTool` (`company_enrich_tool`): Start a Company Enrich for 1-200 domains you already have and wait for it. Same company fields, one company per domain, with no fit score (`score` is `None`).
- `CompanyTasksTool` (`company_tasks_tool`): List your Company Search and Company Enrich tasks, newest first, filtered by `type` (`search` or `enrich`), `status` and `metadata`. Free.
- `CompanyTaskControlTool` (`company_task_control_tool`): Pause, resume or cancel a company task (`action`). Free.

> **Billing and timing**: `company_search_tool` and `company_enrich_tool` are billed. Company Search reserves `count` companies (5-200) and Company Enrich one company per domain, at the route's per-company price ([prices](https://desearch.ai/pricing)); you pay only for the companies a task delivers (at least 2 for Company Search and 1 for Company Enrich when it delivers any), and a task that delivers none is free. A task runs in the background for minutes. The tools wait up to `timeout` seconds (default 300), then return the companies delivered so far with the `task_id`. Call either tool again with only `task_id` to keep waiting: that is free and does not start a new task. `company_tools` is kept out of `search_tools`, so `create_search_agent` never starts a billed task.

#### Monitor Tools
The `monitor_tools` group contains the Monitors tools. A monitor reruns `query` on a schedule (`interval`: `1h`, `3h`, `6h`, `12h`, `1d` or `7d`) over its `sources` (`web`, `x`, `reddit`, `youtube`, `threads`, `tiktok`, `instagram`, `facebook`, `meta_ads`; default `web` and `x`). Each run searches the time since the previous run, drops everything the monitor has seen before, screens the rest with an AI classifier for relevance to `query` and to each of `criteria` (up to 5), and keeps the items whose `score` (0-100, the lowest of those answers) reaches `min_score` (default 60).

A source can also watch up to 10 accounts instead of searching: `x_accounts` (`OpenAI`, `@OpenAI` or an x.com URL), `reddit_subreddits` (`LocalLLaMA`, `r/LocalLLaMA` or a URL), `youtube_channels` (`@handle`, a `UC…` channel ID or a URL), `tiktok_accounts` (`openai`, `@openai` or a URL) and `threads_accounts` (`zuck`, `@zuck` or a URL). Each run then reads every account's newest posts, and `query` and `criteria` still decide what is kept; `min_score: 0` keeps every new post. With X accounts, `x_count` is not used (a timeline page is 20 posts) and `x_lang` and `x_min_likes` still filter. `tiktok_count` (10, 20 or 30, default 10) is per account. The tools send names as given; the API cleans them, returns them cleaned (`OpenAI`, `LocalLLaMA`, `@OpenAI` or `UC…`, `openai`, `zuck`) and answers 422 for a name it cannot read (an X username is at most 15 characters). An account that cannot be read in a run is listed in that run's `stats.sources.<source>.accounts_failed`.

TikTok and Instagram can also watch up to 10 hashtags each: `tiktok_hashtags` and `instagram_hashtags`, in any script (`aversi`, `#ავერსი`, or a tag URL such as `https://www.tiktok.com/tag/aversi` or `https://www.instagram.com/explore/tags/aversi/`). A hashtag feed lists popular as well as recent posts from any account, so a monitor finds posts outside the accounts it watches; with accounts or hashtags, a source reads those feeds instead of searching by keyword. Each run reads `tiktok_count` posts per TikTok hashtag and up to 12 per Instagram hashtag. The API returns hashtags without `#` and answers 422 for one it cannot read. An item read from a hashtag has `data.hashtag`, and a hashtag that cannot be read is listed in the run's `stats.sources.<source>.hashtags_failed`. Facebook has no organic hashtag feed; for ads by keyword use `meta_ads`.

TikTok search has no working date filter and ranks by relevance, so a TikTok keyword monitor finds a new video only when it reaches TikTok's top results. To follow creators or a tag, watch their accounts (`tiktok_accounts`) or the hashtag (`tiktok_hashtags`). Threads keyword search returns loosely matched posts, so `threads_accounts` work better there too.

Instagram and Facebook have no keyword search for posts, so those sources only read watched feeds: `instagram_accounts` (`openai`, `@openai` or a profile URL; up to 12 posts per account per run) or `instagram_hashtags`, and `facebook_pages` (`NASA`, a numeric page ID or a page URL, with `profile.php?id=…` and `/pages/Name/123…` cleaned to the ID; up to 10 posts per page per run). A monitor with `instagram` in `sources` needs at least one account or hashtag, and one with `facebook` at least one page. The tools leave that rule to the API, which answers 422 on create, and on update when the monitor as changed would have none (for example `facebook_pages: []` while `facebook` is a source); the tools return that as an `HTTP 422` error. `meta_ads` searches the Meta Ad Library by keyword, `meta_ads_count` (10, 20 or 30, default 10) records per run. Its search matches keywords, so the `query` of a monitor with `meta_ads` should be a brand, product or short phrase (`ChatGPT`); a sentence finds nothing. An ad is new the first time a run sees it if it started within the last 30 days. For ads by keyword use `meta_ads`, not `facebook`.
- `MonitorCreateTool` (`monitor_create_tool`): Create a monitor. Takes `query`, `name`, `criteria`, `sources`, `interval`, `min_score`, `summary` (a short digest per run), the per-source options (`web_count` 10/15/20, `x_count` 1-100, `x_lang`, `x_min_likes`, `tiktok_count` 10/20/30, `meta_ads_count` 10/20/30) and watched accounts (`x_accounts`, `reddit_subreddits`, `youtube_channels`, `tiktok_accounts`, `threads_accounts`, `instagram_accounts`, `facebook_pages`) and hashtags (`tiktok_hashtags`, `instagram_hashtags`), `webhook_url` and `webhook_events`, `metadata` (up to 20 labels) and `run_now` (default true: the first run is queued at once). With a webhook it returns `webhook_secret`, only this once.
- `MonitorEstimateTool` (`monitor_estimate_tool`): The most one run (`max_run_cost_usd`) and one month of runs can cost for these sources, options (watched accounts and hashtags included), summary and interval, with a `breakdown` per source, `screening` and `summary`. It prices `instagram` or `facebook` without accounts or hashtags as one account. Free.
- `MonitorsTool` (`monitors_tool`): List your monitors (filter by `status`, page with `cursor`), or read one by `monitor_id`. Free. A monitor's `webhook` carries `last_delivery`, its newest delivery as the API returns it (`event`, `status` `pending`, `delivered` or `failed`, `attempts`, `status_code`, `error`, `created_at`, `next_attempt_at`, `delivered_at`), or `null` before the first delivery. `pending` waits for its first or next attempt, with the last answer in `status_code` and `error`; `failed` used every retry.
- `MonitorUpdateTool` (`monitor_update_tool`): Change any setting. Only the settings you set change; the per-source options and the webhook events are merged with the current ones. A new account or hashtag list replaces that source's watched accounts or hashtags, and `[]` stops watching (for `instagram` and `facebook` the API answers 422 while the source has nothing left to watch). A new `webhook_url` gets a new `webhook_secret`; `remove_webhook` removes the webhook. Free.
- `MonitorControlTool` (`monitor_control_tool`): `action` `pause`, `resume`, `trigger` (one billed run now; HTTP 409 while a run is queued or running), `delete`, `test_webhook` (a signed test event; reports what your endpoint answered) or `rotate_webhook_secret`. All but `trigger` are free.
- `MonitorRunsTool` (`monitor_runs_tool`): List a monitor's runs, or read one by `run_id` with the items it kept. Per source, `stats` says how many items were fetched, new and kept, and which watched accounts and hashtags failed (`accounts_failed`, `hashtags_failed`); `stats.screen` is `ai` (screened by an AI classifier) or `keyword` (the fallback). With `run_id`, `timeout` waits for the run to end. Free.
- `MonitorItemsTool` (`monitor_items_tool`): Everything a monitor kept, newest first, filtered by `source` and paged with `cursor`. Each item has its url, title, text (cut to 500 characters), author, `published_at`, `score`, the screening `signals` (`relevant`, `new_info`, and `criteria`, the lowest answer across them, when the monitor has criteria) and source `data`: X `likes`, `reposts`, `replies`, `views`; Reddit `subreddit`, `score`, `comments`; YouTube `channel`, `views`, `thumbnail`; TikTok `views`, `likes`, `comments`, `shares`, `saves`, `duration_s`; Threads `likes`; Instagram `likes`, `comments`, `views`, `media_type`; Facebook `reactions`, `comments`, `shares`, `views`; Meta ads `page`, `active`, `cta`, `link` (the landing page), `platforms`, `thumbnail`, with the ad headline as `title` and `url` the Ad Library record; web `highlights`, `domain`. TikTok, Instagram and Facebook items also have `post_id` (the ID the API's `GET /desearch/social/post` gives the same post; an Instagram shortcode), `type` (`video`, `image`, `carousel`, `reel`, …), `hashtags`, `mentions` (TikTok and Instagram), `media` (links to the post's videos and pictures) and `thumbnail`; media and thumbnail links expire after a few days. The metrics are the post's when the run found it; `GET /desearch/social/post?url=<link>` reads them again (`author=true` adds the author's followers). An item read from a watched account also has `data.account`, and one read from a watched hashtag `data.hashtag`. `source` filters by any of the sources. Free.

> **Billing**: creating, reading, changing, pausing and deleting monitors is free. Each run is billed: its searches at their list price, AI screening at $0.05 per 1,000 items screened, and the summary's LLM usage plus 50% (when `summary` is on). A run holds `max_run_cost_usd` on the balance and bills only what it used; a run with no new items still pays for its searches. A watched account is one feed read per run at that feed's list price: an X account about $0.003 (a 20-post timeline), a subreddit, YouTube channel or Threads account $0.00207, TikTok $0.00207 per post (search, account or hashtag), an Instagram account or hashtag $0.02484 (up to 12 posts at the per-post price) and a Facebook page $0.0207 (up to 10 posts). Meta ads are `meta_ads_count` Ad Library records: 10 is $0.0207, 30 is $0.0621. A run the balance cannot hold fails and pauses the monitor (`paused_reason` `insufficient_credits`) until you resume it. `monitor_tools` is kept out of `search_tools`, so `create_search_agent` never creates a monitor.

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
# sort="recent" puts recently started ads first (the default is impressions).
recent = FacebookTool().invoke({"action": "search", "query": "running shoes", "sort": "recent"})
```

#### Using the YouTube, Reddit and Threads Tools
```python
from langchain_desearch import RedditTool, ThreadsTool, YouTubeTool

youtube = YouTubeTool()
videos = youtube.invoke({"action": "search", "query": "bittensor", "type": "videos"})
video_id = videos["results"][0]["id"]
transcript = youtube.invoke({"action": "transcript", "video_id": video_id, "language": "en"})
print(transcript["text"][:200])

# Page through a channel's uploads.
page = youtube.invoke({"action": "channel_videos", "handle": "@MrBeast", "sort": "popular"})
if page["next_cursor"]:
    page = youtube.invoke({
        "action": "channel_videos", "handle": "@MrBeast", "sort": "popular",
        "cursor": page["next_cursor"],
    })

reddit = RedditTool()
top = reddit.invoke({"action": "subreddit_posts", "name": "AskReddit", "sort": "top", "timeframe": "week"})
post_id = top["results"][0]["id"]
tree = reddit.invoke({"action": "comments", "post_id": post_id})
for comment in tree["results"]:
    print(comment["author"], len(comment["replies"]))

# Threads returns one page; a post comes with its replies.
post = ThreadsTool().invoke({"action": "post", "id": "https://www.threads.com/@zuck/post/DK-BydcJHkF"})
print(post["text"], len(post["comments"]))
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

#### Using Company Search
```python
from langchain_desearch.tools import (
    CompanyEnrichTool,
    CompanySearchTool,
    CompanyTasksTool,
)
from dotenv import load_dotenv
load_dotenv()

# Start a billed Company Search and wait up to 5 minutes for it.
companies = CompanySearchTool().invoke({
    "query": "Series A fintech startups in Berlin building payment infrastructure",
    "count": 25,
    "countries": ["DE"],
    "require": ["email"],
    "metadata": {"campaign": "q4-outbound"},
    "idempotency_key": "q4-berlin-fintech",
    "timeout": 300,
})
print(companies["status"], len(companies["companies"]))
for company in companies["companies"]:
    print(company["domain"], company.get("score"), company.get("primary_email"))

# Still running? Keep waiting on the same task (free, nothing new is billed).
if not companies["done"]:
    companies = CompanySearchTool().invoke({"task_id": companies["task_id"], "timeout": 300})

# Contacts and leaders for domains you already have (billed, one company per domain).
enriched = CompanyEnrichTool().invoke({"domains": ["acme.com", "https://www.beta.io/about"]})

# Your company tasks for one campaign (free).
tasks = CompanyTasksTool().invoke({"metadata": {"campaign": "q4-outbound"}})
```

`company_search_tool` also takes `min_score`, `exclude` (domains or URLs to leave out, up to 1000), `countries` (ISO codes or names, up to 20) and `require` (`email`, `phone`, `linkedin`, `leader`, `leader_email`: a company without all of them is not delivered or billed). Both start tools take `roles` (`ceo`, `founder`, `cto`, `cfo`, `coo`, `cmo`, `president`, `owner`, `managing_director`; default `ceo`, `founder`, `cto`, `cfo`), `infer_emails` and `verify_emails` (both default true), `metadata` (your own labels) and `idempotency_key`. Values the API would reject are refused before any request is sent, so nothing is billed.

Each company has `enrichment` with a `status` (`partial` while any stage, such as leaders, inferred emails or email checks, is still `pending`, then `complete`) and each stage. A stage cut by the task's time limit is `skipped`, with an `enrichment.errors` entry and the task's `enrichment_time_limit` warning. The tools read every company again on each call, so they return the latest version. A finished task has `completion_reason`, `coverage_limited` (when a budget stopped it before `count`) and `warnings`, such as `planning_fallback`, `ai_scoring_unavailable`, `enrichment_time_limit` or `linkedin_verification_unavailable`. A People Search `task_id` does not work on the company tools, and a company `task_id` does not work on `people_search_tool`.

To give an agent the Company tools, pass `company_tools` from `langchain_desearch.search_tools` to your agent. Like the other tools, they set `return_direct=True`; create them with `CompanySearchTool(return_direct=False)` if the agent should read the result and keep going.

> Company Search calls the `/desearch/companies` routes over HTTP with aiohttp, because the Company Search methods in `desearch-py` arrive in 1.8.0, which is not on PyPI yet. The tools will switch to the SDK once it is published.

#### Using Monitors
```python
from langchain_desearch.tools import (
    MonitorControlTool,
    MonitorCreateTool,
    MonitorEstimateTool,
    MonitorItemsTool,
    MonitorRunsTool,
)
from dotenv import load_dotenv
load_dotenv()

# What can it cost? (free)
estimate = MonitorEstimateTool().invoke({"sources": ["web", "x", "reddit"], "interval": "1d"})
print(estimate["max_run_cost_usd"], estimate["max_monthly_cost_usd"])

# Create it (free); the first run is queued at once and billed.
monitor = MonitorCreateTool().invoke({
    "query": "Bittensor subnet launches and updates",
    "criteria": ["Announces a new subnet, release or partnership"],
    "sources": ["web", "x", "reddit"],
    "interval": "1d",
    "webhook_url": "https://example.com/hooks/desearch",
    "metadata": {"team": "growth"},
})
secret = monitor["webhook_secret"]  # returned only once: store it

# Everything it kept so far, newest first (free).
items = MonitorItemsTool().invoke({"monitor_id": monitor["monitor_id"], "source": "x"})
for item in items["items"]:
    print(item["score"], item.get("url"), item.get("text"))

# Run it again now (billed) and wait up to 2 minutes for the run.
run = MonitorControlTool().invoke({"monitor_id": monitor["monitor_id"], "action": "trigger"})
run = MonitorRunsTool().invoke(
    {"monitor_id": monitor["monitor_id"], "run_id": run["run_id"], "timeout": 120}
)
print(run["status"], run.get("item_count"), run.get("cost_usd"))

# Watch accounts instead of searching; min_score 0 keeps every new post.
watch = MonitorCreateTool().invoke({
    "query": "AI model releases",
    "sources": ["x", "tiktok"],
    "x_accounts": ["OpenAI", "@AnthropicAI"],
    "tiktok_accounts": ["openai"],
    "min_score": 0,
    "interval": "6h",
})

# Watch hashtags (any script): posts from any account. Instagram needs no accounts.
tags = MonitorCreateTool().invoke({
    "query": "Aversi pharmacies: promotions, reviews and complaints",
    "sources": ["tiktok", "instagram"],
    "tiktok_hashtags": ["#ავერსი", "https://www.tiktok.com/tag/aversi"],
    "instagram_hashtags": ["aversi", "ავერსი"],
    "min_score": 0,
})
print(tags["options"]["tiktok"]["hashtags"])  # ['ავერსი', 'aversi']

# Instagram and Facebook read only watched accounts; Meta ads match a keyword query.
brand = MonitorCreateTool().invoke({
    "query": "ChatGPT",
    "sources": ["instagram", "facebook", "meta_ads", "threads"],
    "instagram_accounts": ["openai"],
    "facebook_pages": ["https://www.facebook.com/profile.php?id=100064869009429"],
    "threads_accounts": ["@zuck"],
    "meta_ads_count": 20,
    "webhook_url": "https://example.com/hooks/desearch",
    "run_now": False,
})
print(brand["options"]["facebook"], brand["webhook"]["last_delivery"])  # {'pages': ['100064869009429']} None
```

Each webhook delivery is a POST with `Desearch-Event`, `Desearch-Delivery` and `Desearch-Signature: t=<unix>,v1=<hex>` headers. `v1` is the HMAC-SHA256 of `"<t>.<raw body>"` keyed with `webhook_secret`; reject a `t` older than 5 minutes and compare in constant time (desearch-py 1.13.0 adds a helper for it). A delivery that does not get a 2xx within 10 seconds is retried after 1, 5, 30, 120 and 360 minutes, and after 5 deliveries in a row fail the monitor pauses with `paused_reason` `webhook_failing`. `monitors_tool` shows the newest delivery in `webhook.last_delivery` (`pending` while it waits for a retry). `test_webhook` sends a signed `monitor.test` event; it is not a delivery and leaves `last_delivery` unchanged.

To give an agent the Monitor tools, pass `monitor_tools` from `langchain_desearch.search_tools` to your agent.

> Monitors call the `/desearch/monitors` routes over HTTP with aiohttp, like People Search and Company Search, because the Monitors methods in `desearch-py` arrive in 1.13.0, which is not on PyPI yet. The tools will switch to the SDK once it is published.

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

`scripts/e2e-youtube-reddit-threads.py` calls every `youtube_tool`, `reddit_tool` and `threads_tool` action once (one through `ainvoke`) and follows one cursor page on each paged action, about 22 requests at $0.00207 each. `--base-url` and `--key-env` point it at another deployment:
```bash
DESEARCH_API_KEY=... python scripts/e2e-youtube-reddit-threads.py
python scripts/e2e-youtube-reddit-threads.py --base-url https://api-dev.desearch.ai --key-env DESEARCH_DEV_API_KEY
```

`scripts/e2e-monitors.py` creates one monitor without a first run, calls every free Monitors action once (read, list, estimate, update, pause, resume, runs, items, webhook test, secret rotation, webhook removal) and deletes the monitor at the end. Everything it calls is free; `--trigger` also runs the monitor once and waits for the run (billed, about $0.002-0.02). `--with-accounts` also checks watched accounts and the TikTok source, then the Instagram, Facebook and Meta ads sources and Threads accounts: cleaned names, a 422 for instagram or facebook without accounts (on create and on update), the estimate's breakdown per source, `webhook.last_delivery` null on a new monitor, and items by `source` `meta_ads`. Those monitors are created with `run_now: false`, never triggered, and deleted at the end, also when a check fails:
```bash
python scripts/e2e-monitors.py --base-url https://api-dev.desearch.ai --key-env DESEARCH_DEV_API_KEY [--with-accounts]
```

## Project Structure

```
langchain_desearch/
├── langchain_desearch/
│   ├── __init__.py
│   ├── tools.py
│   ├── people.py
│   ├── social.py
│   ├── companies.py
│   ├── monitors.py
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