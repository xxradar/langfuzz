
# LangFuzz: Red Teaming for Language Models
LangFuzz is a tool designed to perform red teaming on language model applications. It generates pairs of similar questions and compares the responses to identify potential failure modes in chatbots or other language model-based systems.

## Installation
To install LangFuzz, use pip:

```
pip install langfuzz
```

## Usage

### Step 1: define a model file
First, define a model file that calls your model. This file should expose an async function called `call_model` that takes in a string and returns a string. An example file is found in [call_model.py](call_model.py). Example model file:

```python
from langgraph_sdk import get_client

client = get_client(url="https://chat-langchain-harrison-5e1205077f2c57788c506fd71cf3b3a0.default.us.langgraph.app")
assistant_id = "chat"


async def call_model(question: str) -> str:
    input_messages = {"messages": [{"role": "user", "content": question}]}
    stateless_run_result = await client.runs.wait(
        None,
        assistant_id,
        input=input_messages,
    )
    return stateless_run_result['messages'][-1]['content']
```

### Step 2: define a configuration file

Next, you need to define a configuration file. 

The `config.json` file should contain the following keys:
- `chatbot_description`: A description of the chatbot being tested.
- `model_file`: Path to the Python file containing the call_model function

Example config.json:
```
{
    "chatbot_description": "Chat over LangChain Docs",
    "model_file": "call_model.py"
}
```

It may optionally contain other keys (see [Options](#options) below).

### Step 4: set any enviornment variables

This requires several environment variables to run:

- `OPENAI_API_KEY`: required for accessing OpenAI model
- [OPTIONAL] `LANGSMITH_API_KEY`: required for sending generated data to a LangSmith dataset.

```
export OPENAI_API_KEY=...
export LANGSMITH_API_KEY=...
```

### Step 3: run the red teaming

To run the red teaming process, use the following command:
```
langfuzz config.json [options]
```

## Options

- `--dataset_id`: ID of the dataset to use (optional)
- `--n`: Number of questions to generate (default: 10)
- `--max_concurrency`: Maximum number of concurrent requests to the model (default: 10)
- `--n_prefill_questions`: Number of questions to prefill the dataset with (default: 10)
- `--max_similarity`: Maximum similarity score to accept (default: 10)
- `-p`, `--persistence-path`: Path to the persistence file (optional)

These options can additionally be provided as part of the configuration file.

## How It Works

The tool generates pairs of similar questions based on the provided chatbot description.
It sends these questions to the chatbot and compares the responses.
A judge model evaluates the similarity of the responses on a scale of 1-10.
Results with similarity scores below the max_similarity threshold are presented for review.
Users can choose to add the questions to a dataset for further analysis or training.
Persistence
If a persistence path is provided, the tool will save generated questions and dataset information between runs. This allows for continuous red teaming sessions without duplicating questions.
