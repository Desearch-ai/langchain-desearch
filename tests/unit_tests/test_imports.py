from langchain_desearch import __all__  # type: ignore[import-not-found, import-not-found]

EXPECTED_ALL = [
    "DesearchTool",
    "BasicWebSearchTool",
    "BasicTwitterSearchTool",
    "AIWebLinksSearchTool",
    "AIXLinksSearchTool",
    "SearchApiTool",
    "WebExtractTool",
    "SearchTweetsByUserTool",
    "FetchTwitterTrendsTool",
    "TikTokTool",
    "FacebookTool",
    "InstagramTool",
    "PeopleSearchTool",
    "PeopleProfilesTool",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
