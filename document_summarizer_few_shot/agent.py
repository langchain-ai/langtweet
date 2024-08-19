from typing import TypedDict, Literal, Optional, Any, Dict

from langchain_core.pydantic_v1 import BaseModel, Field


from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from document_summarizer_few_shot.loading import get_content
from document_summarizer_few_shot.utils.fewshot import search_similar_examples

class SoundsLikeHarrison(BaseModel):
    sounds_like_harrison: bool = Field(description="whether the tweet sounds like harrison chase")
    critique: Optional[str] = Field(description="If it does not sound like Harrison, instructions on how to sound more like Harrison")


class DirectSummarizerState(TypedDict):
    url: str
    content: str
    facts: list[str]
    tweet: str
    attempts: int
    critique: SoundsLikeHarrison

class GraphInput(TypedDict):
    url: str

class GraphOutput(TypedDict):
    tweet: str


def get_contents(state, config):
    url = state['url']
    content = get_content(url)
    return {"content": content}


class NotEnoughInfo(TypedDict):
    call: bool

class Facts(TypedDict):
    facts: list[str]


convert_to_facts_prompt = """You are Harrison Chase. You run a company focused on the latest AI technologies. \
Your followers love hearing your thoughts on the latest advancements in AI. You are optomistic. \
You are particularly interested in new AI methods and techniques, and highlighting interesting \
use cases of your LLM frameworks, LangChain LangGraph and LangSmith.

I will give you an article. Extract a bulleted list of key points made or statements made in it.

If there is not enough information to generate a good list of info, then call the `NotEnoughInfo` tool.
Otherwise, ONLY generate a list of facts Using the `Facts` tool.

Here is the article:

{article}"""

def convert_to_facts(state, config):
    print("facts state: ", state)
    content = state.get("content")
    if not content: 
        raise ValueError("No content provided")

    model = ChatOpenAI(model_name="gpt-4o-mini").bind_tools([NotEnoughInfo, Facts])
    
    response = model.invoke(
        [
            {
                "role": "user",
                "content": convert_to_facts_prompt.format(article=content)
            },
        ]
    )

    if len(response.tool_calls) == 0:
        return {"facts": None}
    else:
        tool_call = response.tool_calls[0]
        if tool_call['name'] == "Facts":
            return {"facts": tool_call['args']['facts']}


base_prompt = """You are Harrison Chase. You tweet highlighting information related to LangChain, your LLM company.
You use emojis. You use exclamation points but are not overly enthusiastic. You never use hashtags.
You sometimes make spelling mistakes. You are not overly formal. You are not "salesy". You are nice.
Don't be too cheesy.

You summarize lists of facts derived from articles into tweets. Here are past examples:
"""

def write_tweet(state, config):
    print("tweets state: ", state)
    facts = state.get("facts")
    if not facts: 
        raise ValueError("No facts provided")
    
    from langsmith import traceable, wrappers
    from openai import Client
    from langsmith import traceable, wrappers

    openai = wrappers.wrap_openai(Client())

    # TODO: FILL IN WITH YOUR BASE PROMPT

    examples = search_similar_examples(
        dataset_id="6755b475-ae3a-4515-b070-b02821d88418",
        inputs_dict={
            "facts": facts
        },
        limit=5
    )["examples"]
    
    messages = [SystemMessage(content=base_prompt)]
    for e in examples:
        messages.append(HumanMessage(content="".join(["<fact> " + fact + "</fact>" for fact in e['inputs']['facts']])))
        messages.append(AIMessage(content=e['outputs']['tweet']))

    messages.append(HumanMessage(content="".join(["<fact> " + fact + "</fact>" for fact in facts])))

    chain = ChatOpenAI() | StrOutputParser()
    chain = ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620") | StrOutputParser()
    
    tweet = chain.invoke(messages)

    attempts = state.get("attempts", 0) or 0
    return {"tweet": tweet, "attempts": attempts + 1}


def _check_if_enough_info(state: DirectSummarizerState) -> Literal["write_tweet", END]:
    if "facts" in state:
        return "write_tweet"
    else:
        return END


summarizer_workflow = StateGraph(DirectSummarizerState, input=GraphInput, output=GraphOutput)
summarizer_workflow.add_node(get_contents)
summarizer_workflow.add_node(convert_to_facts)
summarizer_workflow.add_node(write_tweet)

summarizer_workflow.set_entry_point("get_contents")
summarizer_workflow.add_edge("get_contents", "convert_to_facts")
summarizer_workflow.add_conditional_edges("convert_to_facts", _check_if_enough_info)
summarizer_workflow.add_edge("write_tweet", END)


summarizer_graph = summarizer_workflow.compile()

class DirectSummarizerState(TypedDict):
    url: str
    content: str
    tweet: str


base_prompt_article = """You are Harrison Chase. You tweet highlighting information related to LangChain, your LLM company.
You use emojis. You use exclamation points but are not overly enthusiastic. You never use hashtags.
You sometimes make spelling mistakes. You are not overly formal. You are not "salesy". You are nice.

When given an article, write a tweet about it. Make it relevant and specific to the article at hand.

Pay attention to the examples below. These are good examples. Generate future tweets in the style of the tweets below."""
def write_tweet_from_article(state, config):
    print("tweets state: ", state)

    from langsmith import traceable, wrappers
    from openai import Client
    from langsmith import traceable, wrappers

    examples = search_similar_examples(
        dataset_id="f6320c32-a6ff-46f1-9cc4-1b65e3e14a07",
        inputs_dict={
            "content": state['content']
        },
        limit=5
    )["examples"]

    messages = [SystemMessage(content=base_prompt_article)]
    for e in examples:
        messages.append(HumanMessage(content=e['inputs']['content']))
        messages.append(AIMessage(content=e['outputs']['tweet']))

    messages.append(HumanMessage(content=state['content']))

    chain = ChatOpenAI(model_name="gpt-4o") | StrOutputParser()

    tweet = chain.invoke(messages)

    attempts = state.get("attempts", 0) or 0
    return {"tweet": tweet, "attempts": attempts + 1}

tweet_workflow = StateGraph(DirectSummarizerState, input=GraphInput, output=GraphOutput)
tweet_workflow.add_node(get_contents)
tweet_workflow.add_node(write_tweet_from_article)

tweet_workflow.set_entry_point("get_contents")
tweet_workflow.add_edge("get_contents", "write_tweet_from_article")
tweet_workflow.add_edge("write_tweet_from_article", END)


tweet_graph = tweet_workflow.compile()


# this is the graph making function that will decide which graph to
# build based on the provided config
def make_graph(config):
    raw_article = config.get("raw_article", {}).get("raw_article", True)
    # route to different graph state / structure based on the user ID
    if raw_article:
        return tweet_graph
    else:
        return summarizer_graph


