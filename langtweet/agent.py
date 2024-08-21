from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langsmith import client


from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from langtweet.loading import get_content

langsmith_client = client.Client()


class GraphInput(TypedDict):
    url: str


class GraphOutput(TypedDict):
    tweet: str


def get_contents(state):
    url = state["url"]
    content = get_content(url)
    return {"content": content}


class DirectSummarizerState(TypedDict):
    url: str
    content: str
    tweet: str


base_prompt = """You are Harrison Chase. You tweet highlighting information related to LangChain, your LLM company.
You use emojis. You use exclamation points but are not overly enthusiastic. You never use hashtags.
You sometimes make spelling mistakes. You are not overly formal. You are not "salesy". You are nice.

When given an article, write a tweet about it. Make it relevant and specific to the article at hand.

Pay attention to the examples below. These are good examples. Generate future tweets in the style of the tweets below."""


def write_tweet_from_article(state: DirectSummarizerState):

    examples = langsmith_client.similar_examples(
        {"content": state["content"]},
        dataset_id="f6320c32-a6ff-46f1-9cc4-1b65e3e14a07",
        limit=5,
    )

    messages = [SystemMessage(content=base_prompt)]
    for e in examples:
        messages.append(HumanMessage(content=e.inputs["content"]))
        messages.append(AIMessage(content=e.outputs["tweet"]))

    messages.append(HumanMessage(content=state["content"]))

    chain = ChatOpenAI(model_name="gpt-4o")

    tweet = chain.invoke(messages)
    return {"tweet": tweet.content}


tweet_workflow = StateGraph(DirectSummarizerState, input=GraphInput, output=GraphOutput)
tweet_workflow.add_node(get_contents)
tweet_workflow.add_node(write_tweet_from_article)

tweet_workflow.set_entry_point("get_contents")
tweet_workflow.add_edge("get_contents", "write_tweet_from_article")
tweet_workflow.add_edge("write_tweet_from_article", END)


tweet_graph = tweet_workflow.compile()
