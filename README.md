# LangChain Desearch Integration

This project integrates the Desearch API with LangChain tools to enable various search and data-fetching functionalities, such as web searches, Twitter data retrieval, and AI-powered searches.

## Features

- **Grouped Tools**:
  - **Search Tools**: General-purpose search tools for AI, web, and Twitter searches.
  - **Twitter Tools**: Tools specifically for Twitter-related operations.

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
- `FetchRetweetsByPostTool`: Retrieve retweets of a specific post.
- `FetchTwitterUserTool`: Get detailed information about a Twitter user.

### Examples

#### Using Tools
```python
from langchain_desearch.tools import DesearchTool, BasicWebSearchTool, BasicTwitterSearchTool

# Example 1: Using DesearchTool
tool = DesearchTool()
result = tool._run(
    prompt="Bittensor",
    tool="desearch_web",
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
    return desearch_tool._run(prompt=prompt, tool="desearch_web", model="NOVA")

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
pytest tests/test_tools.py
```

#### Real API Tests
Run the real tests to verify the tools' functionality with the Desearch API:
```bash
pytest tests/test_tools_real.py
```

> **Note**: Ensure you have a valid `DESEARCH_API_KEY` in your `.env` file before running real tests.

## Project Structure

```
langchain_desearch/
├── langchain_desearch/
│   ├── __init__.py
│   ├── tools.py
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