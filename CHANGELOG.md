# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - unreleased

> **Do not publish 1.2.0 before desearch-py 1.8.0 is on PyPI.** This release
> requires `desearch-py>=1.8.0,<2.0.0`, the first desearch-py with the TikTok,
> Facebook, Instagram and Search API methods, and desearch-py 1.8.0 is not on
> PyPI yet. Until it is, `pip install langchain-desearch==1.2.0` cannot
> resolve its dependencies.

### Added

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
