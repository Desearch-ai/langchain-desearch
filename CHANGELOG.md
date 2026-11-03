# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - unreleased

> **Waits for the Company Search API release that adds `max_cost_usd` and
> `continue_from` (desearch-public-api#624).** Until then the API refuses a
> search that sends either one with HTTP 422. The company tools call the
> `/desearch/companies` routes over HTTP, so this release needs no new
> desearch-py; it has the same requirement as 1.3.0 (`desearch-py>=1.11.0`,
> not on PyPI yet). If 1.3.0 is still unpublished, 1.4.0 ships both.

### Added

- `company_search_tool` takes `max_cost_usd`, a search budget in USD (above 0,
  at most 100). Without it, a search stops once it costs more than the
  companies it found pay for (`coverage_limited`, `cost_limit`). With it, the
  whole budget is reserved, the search keeps going until it delivers `count`
  or spends the budget, and its search effort is billed up to the budget even
  when it finds few companies or none. The field description tells the agent
  to set it only when the user agreed to that budget. A budget below what
  `count` reserves is refused by the API (HTTP 422, with the minimum).
- `company_search_tool` takes `continue_from`, the `task_id` of an ended
  Company Search: a new search that leaves out every company that one
  delivered or held for review. It must be a task_id (UUID) and cannot be
  combined with `task_id`, which waits on a search instead.
- Both are sent only when given. The compact task returns them as
  `max_cost_usd` and `continue_from` (from `task.request`).
- After a search that stopped before `count` (`coverage_limited`),
  `next_step` offers a search with `continue_from`, says when the search
  spent its budget or hit the cost limit, and repeats that `max_cost_usd`
  needs the user's agreement. Company Enrich gets no such offer.
- The `query` description says how to restrict a search to named companies:
  name them with their websites.

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
