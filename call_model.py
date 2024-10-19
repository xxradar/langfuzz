from langgraph_sdk import get_client

client = get_client(url="https://chat-langchain-harrison-5e1205077f2c57788c506fd71cf3b3a0.default.us.langgraph.app")
assistant_id = "chat"


async def call_model(inputs):
    input_messages = {"messages": [{"role": "user", "content": inputs}]}
    stateless_run_result = await client.runs.wait(
        None,
        assistant_id,
        input=input_messages,
    )
    return stateless_run_result['messages'][-1]['content']
