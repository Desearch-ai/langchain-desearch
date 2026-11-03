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
    YouTubeTool,
    RedditTool,
    ThreadsTool,
    PeopleSearchTool,
    PeopleProfilesTool,
    CompanySearchTool,
    CompanyEnrichTool,
    CompanyTasksTool,
    CompanyTaskControlTool,
    MonitorCreateTool,
    MonitorEstimateTool,
    MonitorsTool,
    MonitorUpdateTool,
    MonitorControlTool,
    MonitorRunsTool,
    MonitorItemsTool,
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
# profile_posts, comments, hashtag, and trending on TikTok; YouTube, Reddit and
# Threads have their own actions, such as transcript or subreddit_posts).
social_tools = [
    TikTokTool(),
    FacebookTool(),
    InstagramTool(),
    YouTubeTool(),
    RedditTool(),
    ThreadsTool(),
]

# Group People Search tools into a separate list. people_search_tool is billed
# and takes minutes, so it is not part of search_tools.
people_tools = [
    PeopleSearchTool(),
    PeopleProfilesTool(),
]

# Group People Search tools into a separate list. people_search_tool is billed
# and takes minutes, so it is not part of search_tools.
people_tools = [
    PeopleSearchTool(),
    PeopleProfilesTool(),
]

# Group Company Search tools into a separate list. company_search_tool and
# company_enrich_tool are billed and take minutes, so they are not part of
# search_tools.
company_tools = [
    CompanySearchTool(),
    CompanyEnrichTool(),
    CompanyTasksTool(),
    CompanyTaskControlTool(),
]

# Group Monitors tools into a separate list. A monitor runs on a schedule and
# each run is billed (monitor_create_tool queues the first one at once, and
# monitor_control_tool action='trigger' runs one now), so they are not part of
# search_tools.
monitor_tools = [
    MonitorCreateTool(),
    MonitorEstimateTool(),
    MonitorsTool(),
    MonitorUpdateTool(),
    MonitorControlTool(),
    MonitorRunsTool(),
    MonitorItemsTool(),
]
