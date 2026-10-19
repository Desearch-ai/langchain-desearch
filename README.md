# LangChain Desearch Integration

This project integrates the Desearch API with LangChain tools to enable various search and data-fetching functionalities, such as web searches, Twitter data retrieval, and AI-powered searches.

## Features

- **Grouped Tools**:
  - **Search Tools**: General-purpose search tools for AI, web, and Twitter searches.
  - **Twitter Tools**: Tools specifically for Twitter-related operations.
  - **People Tools**: Find people on X, LinkedIn, YouTube and TikTok who match a plain-language description, and browse everyone your searches found.

## Installation

Install the package using pip:

```bash
pip install langchain-desearch
```

## Usage

### Grouped Tools

#### Search Tools
The `search_tools` group contains tools for general-purpose searches:
- `DesearchTool`: Perform AI searches, web link searches, and Twitter post searches.
- `BasicWebSearchTool`: Conduct basic web searches.
- `BasicTwitterSearchTool`: Perform advanced Twitter searches with filters.

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

## Project Structure

```
langchain_desearch/
├── langchain_desearch/
│   ├── __init__.py
│   ├── tools.py
│   ├── people.py
│   ├── search_tools.py
│   ├── agent.py
├── examples/
│   ├── tools.py
│   ├── RAG.py
│   ├── agent.py
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