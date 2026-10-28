from .tools import (
    DesearchTool,
    BasicWebSearchTool,
    BasicTwitterSearchTool,
    AIWebLinksSearchTool,
    AIXLinksSearchTool,
    SearchApiTool,
    WebExtractTool,
    FetchTweetsByUrlsTool,
    FetchTweetsByIdTool,
    FetchLatestTweetsTool,
    FetchTweetsAndRepliesByUserTool,
    FetchRepliesByPostTool,
    FetchRetweetsByPostTool,
    FetchTwitterUserTool,
    SearchTweetsByUserTool,
    FetchTwitterTrendsTool,
    TikTokTool,
    FacebookTool,
    InstagramTool,
    PeopleSearchTool,
    PeopleProfilesTool,
)

# General-purpose search tools: AI search, web and X search, AI link search,
# the Search API and page extraction. create_search_agent uses this list.
search_tools = [
    DesearchTool(),
    BasicWebSearchTool(),
    BasicTwitterSearchTool(),
    AIWebLinksSearchTool(),
    AIXLinksSearchTool(),
    SearchApiTool(),
    WebExtractTool(),
]

# Group Twitter-related tools into a separate list
twitter_tools = [
    BasicTwitterSearchTool(),
    FetchTweetsByUrlsTool(),
    FetchTweetsByIdTool(),
    FetchLatestTweetsTool(),
    FetchTweetsAndRepliesByUserTool(),
    FetchRepliesByPostTool(),
    FetchRetweetsByPostTool(),
    FetchTwitterUserTool(),
    SearchTweetsByUserTool(),
    FetchTwitterTrendsTool(),
    AIXLinksSearchTool(),
]

# One tool per platform; each takes an `action` (search, post, profile,
# profile_posts, comments, hashtag, and trending on TikTok).
social_tools = [
    TikTokTool(),
    FacebookTool(),
    InstagramTool(),
]

# Group People Search tools into a separate list. people_search_tool is billed
# and takes minutes, so it is not part of search_tools.
people_tools = [
    PeopleSearchTool(),
    PeopleProfilesTool(),
]
