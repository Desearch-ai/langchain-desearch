# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - unreleased

> **Do not publish 1.5.0 before the Monitors API (with its TikTok source and
> watched accounts) is on production.** The tools call the Monitors routes
> over HTTP (`monitors.py`), so this release needs no new desearch-py; it keeps
> `desearch-py>=1.11.0,<2.0.0`, which must be on PyPI first (see 1.3.0). The
> version assumes #7 (1.4.0) lands first; if this lands first, it can be 1.4.0.

### Added

- Monitors tools in `monitor_tools`: searches that rerun on a schedule over
  the web, X, Reddit, YouTube and Threads and keep only what is new. They call
  the `/desearch/monitors` routes over HTTP, with the People Search transport.
  - `MonitorCreateTool` (`monitor_create_tool`): create a monitor (`query`,
    `name`, `criteria`, `sources`, `interval`, `min_score`, `summary`,
    `web_count`, `x_count`, `x_lang`, `x_min_likes`, `webhook_url`,
    `webhook_events`, `metadata`, `run_now`). Returns `webhook_secret` once.
  - `MonitorEstimateTool` (`monitor_estimate_tool`): the most one run and one
    month of runs can cost.
  - `MonitorsTool` (`monitors_tool`): list monitors (`status`, `limit`,
    `cursor`) or read one by `monitor_id`.
  - `MonitorUpdateTool` (`monitor_update_tool`): change settings. Web and X
    options and webhook events are merged with the current ones, because
    `PATCH` replaces each object whole; `remove_webhook` removes the webhook.
  - `MonitorControlTool` (`monitor_control_tool`): `pause`, `resume`,
    `trigger` (one billed run), `delete`, `test_webhook`,
    `rotate_webhook_secret`.
  - `MonitorRunsTool` (`monitor_runs_tool`): list runs, or read one run with
    its items; `timeout` waits for it to end.
  - `MonitorItemsTool` (`monitor_items_tool`): everything a monitor kept,
    newest first, by `source`, paged with `next_cursor`.
- Values the API would reject (query length, more than 5 criteria, unknown
  sources, intervals or events, `web_count` other than 10/15/20, a non-HTTPS
  webhook URL, more than 20 metadata labels) are refused before any request.
- Every result has a `next_step`: where to read what a monitor keeps, how to
  resume a paused one (`paused_reason`), how to wait on a run, and that
  `webhook_secret` is shown only once.
- `scripts/e2e-monitors.py`: a live smoke test of every free Monitors action
  with `--base-url` and `--key-env`; `--trigger` adds one billed run.
- TikTok as a Monitors source (`sources` and the items `source` filter) with
  `tiktok_count` (10, 20 or 30; per account with accounts).
- Watched accounts, up to 10 per source: `x_accounts`, `reddit_subreddits`,
  `youtube_channels` and `tiktok_accounts` on create, estimate and update. A
  source with accounts reads their newest posts each run instead of
  searching; `query` and `criteria` still decide what is kept, and
  `min_score` 0 keeps every new post. Names are sent as given; the API cleans
  them and returns them cleaned (422 for a name it cannot read). On update a
  new list replaces that source's accounts, `[]` stops watching, and every
  other option is kept.
- Descriptions name what a watched account costs per run (X about $0.003,
  a subreddit or YouTube channel $0.00207, TikTok $0.00207 per post), TikTok's
  item `data` (`views`, `likes`, `comments`, `shares`, `duration_s`,
  `thumbnail`), `data.account` on items from a watched account, and a run's
  `stats.sources.<source>.accounts_failed`. They also say that TikTok search
  has no working date filter and ranks by relevance, so a TikTok keyword
  monitor finds a new video only once it reaches TikTok's top results; to
  follow creators, watch their accounts (`tiktok_accounts`).

## [1.3.0] - unreleased

> **Do not publish 1.3.0 before desearch-py 1.11.0 is on PyPI.** This release
> requires `desearch-py>=1.11.0,<2.0.0`, the first desearch-py with the
> YouTube, Reddit and Threads methods, and desearch-py 1.11.0 is not on PyPI
> yet. PyPI's latest langchain-desearch is 1.0.6, so 1.1.0 and 1.2.0 (People
> Search, Company Search, full API coverage) ship for the first time with 1.3.0.

### Added

- `YouTubeTool` (`youtube_tool`), `RedditTool` (`reddit_tool`) and
  `ThreadsTool` (`threads_tool`): one tool per platform with an `action`
  field, covering all 16 YouTube, Reddit and Threads endpoints.
  - YouTube: `search` (`type`, `upload_date`, `sort_by`, `duration`),
    `video`, `transcript` (`language`), `comments` (`order`), `channel`,
    `channel_videos` (`sort`).
  - Reddit: `search` (`sort`, `timeframe`), `subreddit`, `subreddit_posts`
    (`sort`, `timeframe`), `post`, `comments` (a tree with loaded replies).
  - Threads: `search` (`start_date`, `end_date`), `search_users`, `profile`,
    `profile_posts`, `post` (with its replies and thread continuation).
- The YouTube and Reddit list actions page like the other social tools:
  `{"results": [...], "next_cursor": ...}`, with the cursor from the
  `x-desearch-next-cursor` header. Threads returns one page, so its
  `next_cursor` is always `None`. These routes take no `count`.
- Fields with a fixed set of values are checked per action before any
  request (for example Reddit `sort` differs between `search` and
  `subreddit_posts`), `start_date`/`end_date` must be `YYYY-MM-DD`, and a
  leading `r/` is removed from a subreddit `name`.
- The three tools join `social_tools` and are exported from
  `langchain_desearch`.
- `scripts/e2e-youtube-reddit-threads.py`: a live smoke test of the new tools
  with `--base-url` and `--key-env` (for example against api-dev).

### Changed

- Requires `desearch-py>=1.11.0,<2.0.0` (was `>=1.8.0`).

## [1.2.0] - unreleased

> Never published on its own; it ships as part of 1.3.0.

### Added

- Company Search tools in `company_tools` (#4): `CompanySearchTool`
  (`company_search_tool`), `CompanyEnrichTool` (`company_enrich_tool`),
  `CompanyTasksTool` (`company_tasks_tool`) and `CompanyTaskControlTool`
  (`company_task_control_tool`). They call the `/desearch/companies` routes
  over HTTP (see `companies.py`).

- `AIWebLinksSearchTool` (`ai_web_links_search_tool`): web links for a prompt,
  picked by Desearch AI, without a summary (`ai_web_links_search`,
  `POST /desearch/ai/search/links/web`).
- `AIXLinksSearchTool` (`ai_x_links_search_tool`): X posts for a prompt, picked
  by Desearch AI (`ai_x_links_search`, `POST /desearch/ai/search/links/twitter`).
- `SearchTweetsByUserTool` (`search_tweets_by_user_tool`): one X user's posts,
  optionally matching a query (`x_posts_by_user`, `GET /twitter/post/user`).
- `FetchTwitterTrendsTool` (`fetch_twitter_trends_tool`): X trending topics by
  WOEID, Worldwide by default (`x_trends`, `GET /twitter/trends`).
- `WebExtractTool` (`web_extract_tool`): a public page as text or HTML, with
  optional JavaScript rendering (`extract`, `GET /web/extract`).
- `SearchApiTool` (`search_api_tool`): ranked pages, each with the passage that
  best matches the query, with mode, count, date range, highlights and page
  text options (`search_api`, `POST /search`).
- `TikTokTool` (`tiktok_tool`), `FacebookTool` (`facebook_tool`) and
  `InstagramTool` (`instagram_tool`): one tool per platform with an `action`
  field, covering all 19 TikTok, Facebook and Instagram endpoints (search,
  post, profile, profile posts, comments, hashtag, and TikTok trending). Each
  action is checked before any request: a missing target, a field the action
  does not use, or a cursor on an action that does not paginate raises a
  validation error that lists the fields the action takes.
- Social list actions return `{"results": [...], "next_cursor": ...}`. The
  cursor comes from the `x-desearch-next-cursor` response header (desearch-py
  `include_metadata=True`); pass it back unchanged as `cursor` to read the next
  page. It is `None` on the last page.
- A `social_tools` group in `langchain_desearch.search_tools`. The AI links,
  Search API and extract tools join `search_tools`; the X additions join
  `twitter_tools`. Every new tool is exported from `langchain_desearch`.
- `scripts/e2e-full-coverage.py`, a live smoke test of the new tools (needs
  `DESEARCH_API_KEY`; not run by CI; starts a billed People Search only with
  `--with-people`).

### Changed

- Requires `desearch-py>=1.8.0,<2.0.0` (was `>=1.1.0,<2.0.0`).
- Requires Python 3.10 or newer (was 3.9), as desearch-py 1.8.0 does.
- `create_search_agent` now gets seven tools: the three it had plus the AI
  links, Search API and extract tools.

## [1.1.0] - unreleased

### Added

- `PeopleSearchTool` (`people_search_tool`) starts a billed People Search on X,
  LinkedIn, YouTube or TikTok, waits for it and returns compact people.
  Called again with only `task_id`, it keeps waiting without starting a new
  search.
- `PeopleProfilesTool` (`people_profiles_tool`) lists the free profiles library
  with filters and sorting.
- A `people_tools` group, kept out of `search_tools`.

## [1.0.7]

### Fixed

- Migrated the tools to desearch-py 1.1+ (`web_search`, `x_search` and the
  current X endpoints) and pydantic v2; required `desearch-py>=1.1.0,<2.0.0`.
