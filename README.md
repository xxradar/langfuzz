
# LangFuzz: Red Teaming for Language Models
LangFuzz is a tool designed to perform red teaming on language model applications. It generates pairs of similar questions and compares the responses to identify potential failure modes in chatbots or other language model-based systems.

## Installation
To install LangFuzz, use pip:

```
pip install langfuzz
```

## Usage

To run the red teaming process, use the following command:
```
langfuzz config.json [options]
```
## Configuration File
The `config.json` file should contain the following keys:
- `chatbot_description`: A description of the chatbot being tested
- `model_file`: Path to the Python file containing the call_model function
Example config.json:
```
{
    "chatbot_description": "Chat over LangChain Docs",
    "model_file": "call_model.py"
}
```
## Options
- `--dataset_id`: ID of the dataset to use (optional)
- `--n`: Number of questions to generate (default: 10)
- `--max_concurrency`: Maximum number of concurrent requests to the model (default: 10)
- `--n_prefill_questions`: Number of questions to prefill the dataset with (default: 10)
- `--max_similarity`: Maximum similarity score to accept (default: 10)
- `-p`, `--persistence-path`: Path to the persistence file (optional)

## How It Works

The tool generates pairs of similar questions based on the provided chatbot description.
It sends these questions to the chatbot and compares the responses.
A judge model evaluates the similarity of the responses on a scale of 1-10.
Results with similarity scores below the max_similarity threshold are presented for review.
Users can choose to add the questions to a dataset for further analysis or training.
Persistence
If a persistence path is provided, the tool will save generated questions and dataset information between runs. This allows for continuous red teaming sessions without duplicating questions.
