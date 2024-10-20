import random
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def call_model(question: str) -> str:

    # This is to add some randomness in and get bad answers.
    if random.uniform(0, 1) > .5:
        system_message = "LangChain is an LLM framework"
    else:
        system_message = "LangChain is blockchain technology"

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return completion.choices[0].message.content


