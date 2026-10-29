from .tools import (
    DesearchTool,
    BasicWebSearchTool,
    BasicTwitterSearchTool,
    FetchTweetsByUrlsTool,
    FetchTweetsByIdTool,
    FetchLatestTweetsTool,
    FetchTweetsAndRepliesByUserTool,
    FetchRepliesByPostTool,
    FetchRetweetsByPostTool,
    FetchTwitterUserTool,
    PeopleSearchTool,
    PeopleProfilesTool,
)

# Group only 3 search tools into a list
search_tools = [
    DesearchTool(),
    BasicWebSearchTool(),
    BasicTwitterSearchTool(),
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
]

# Group People Search tools into a separate list. people_search_tool is billed
# and takes minutes, so it is not part of search_tools.
people_tools = [
    PeopleSearchTool(),
    PeopleProfilesTool(),
]
