from typing import TypedDict, Literal, Optional, Any, Dict

from langchain_core.pydantic_v1 import BaseModel, Field


from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI

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




def write_tweet(state, config):
    print("tweets state: ", state)
    facts = state.get("facts")
    if not facts: 
        raise ValueError("No facts provided")
    
    from langsmith import traceable, wrappers
    from openai import Client
    from langsmith import traceable, wrappers

    openai = wrappers.wrap_openai(Client())

    @traceable
    def generate_example_prompt(example: Dict[str, Any]):
        # TODO: FILL IN WITH YOUR FEW SHOT PROMPT
        return f"""
            <example>
                <facts>
                    {"".join(["<fact> " + fact + "</fact>" for fact in example['inputs']['facts']])}
                </facts>
                <tweet> {example['outputs']['tweet']} </tweet>
            </example>
            """

    # TODO: FILL IN WITH YOUR BASE PROMPT
    base_prompt = """
        You are Harrison Chase. You tweet highlighting information related to LangChain, your LLM company.
        You use emojis. You use exclamation points but are not overly enthusiastic. You never use hashtags.
        You sometimes make spelling mistakes. You are not overly formal. You are not "salesy". You are nice.

        You summarize lists of facts derived from articles into tweets. Here are past examples:
    """
    examples = search_similar_examples(
        dataset_id="6755b475-ae3a-4515-b070-b02821d88418",
        inputs_dict={
            "facts": facts
        },
        limit=5
    )["examples"]
    
    example_prompt = "\n".join(generate_example_prompt(example) for example in examples)

        
    messages = [
        SystemMessage(
            f"""
                {base_prompt}
                {example_prompt}
            """
        ),
        HumanMessage(
            f"""
                Write a tweet based on the following facts. Twenty words or less:
                {"".join(["<fact>" + fact + "</fact>" for fact in facts])}
            """
        )
    ]


    chain = ChatOpenAI() | StrOutputParser()
    
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

