import warnings

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")

import os
import sys
import pytest
import time  # Add import for simulating delay

# Add the absolute project root directory to sys.path (one level up from tests/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy API key
os.environ["DESEARCH_API_KEY"] = ""


# Dummy implementation for testing.
class DummyDesearch:
    def web_links_search(self, prompt, tools, model, date_filter, streaming):
        time.sleep(2)  # Simulate waiting for search results
        return f"web search result for {prompt}"

    def basic_twitter_search(self, query, model, sort, count):
        return f"twitter search result for {query}"

    def tweets_and_replies_by_user(self, user, query, count):
        return f"tweets and replies for {user}"

    def twitter_replies_post(self, post_id, count, query):
        return f"twitter replies for {post_id}"

    def twitter_retweets_post(self, post_id, count, query):
        return f"twitter retweets for {post_id}"

    def twitter_by_urls(self, urls):
        return f"twitter by urls {urls}"

    def twitter_by_id(self, id):
        return f"twitter by id {id}"

    def tweets_by_user(self, user, query, count):
        return f"tweets by user {user}"

    def basic_web_search(self, query, num, start):
        return f"basic web search for {query}"

    def latest_twits(self, user, count):
        return f"latest twits for {user}"


# Monkey-patch Desearch in the desearch_py package
import desearch_py

desearch_py.Desearch = lambda api_key: DummyDesearch()

# Import the tool to test
from langchain_desearch.tools import DesearchToolTool


def test_web_tool():
    tool = DesearchToolTool()
    input_data = {
        "prompt": "test prompt",
        "tool": "web",
        "model": "NOVA",
        "date_filter": None,
        "streaming": False,
        "query": None,
    }
    result = tool._run(**input_data)
    print(result)  # Print result for debugging; remove if not required.
    assert "web search result" in result


if __name__ == "__main__":
    # Direct call for debugging
    test_web_tool()
    # Then run pytest to collect and run tests if desired.
    import pytest

    pytest.main([__file__])
