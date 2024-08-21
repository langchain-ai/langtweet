import streamlit as st
from langgraph_sdk import get_client
from functools import partial
import asyncio
from langsmith import Client


ls_client = Client()


async def _predict(url):
    client = get_client(url="http://localhost:62715")
    thread = await client.threads.create()
    value = None
    async for chunk in client.runs.stream(
        thread["thread_id"], "tweeter", input={"url": url}, stream_mode="values"
    ):
        if chunk.event == "values":
            value = chunk.data.get("tweet")
            content = chunk.data.get("content")
    return (value, content)


def generate_tweet(url):
    async_func = partial(
        _predict,
        *[url],
    )
    loop = asyncio.new_event_loop()
    contents = loop.run_until_complete(async_func())
    return contents


def call_api(tweet, context):
    ls_client.create_examples(
        inputs=[{"content": context}],
        outputs=[{"tweet": tweet}],
        dataset_id="f6320c32-a6ff-46f1-9cc4-1b65e3e14a07",
    )


st.title("Tweet Generator")

url = st.text_input("Enter a URL:")

if url:
    if "generated_tweet" not in st.session_state:
        t, c = generate_tweet(url)
        print(c)
        st.session_state.generated_tweet = t
        st.session_state.context = c

    tweet = st.text_area(
        "Generated Tweet:",
        value=st.session_state.generated_tweet,
        height=300,
        key="tweet_area",
    )

    if st.button("Give Feedback"):
        # Get the current content of the text area
        current_tweet = st.session_state.tweet_area
        # Send to API
        call_api(current_tweet, st.session_state.context)

st.markdown("---")
st.markdown(
    "Enter a URL above to generate a tweet. You can edit the generated tweet and then give feedback."
)
