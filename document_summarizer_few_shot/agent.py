from typing import TypedDict, Literal, Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from document_summarizer_few_shot.loading import get_content

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
    facts: str


convert_to_facts_system_prompt = """You are Harrison Chase. You run a company focused on the latest AI technologies. \
Your followers love hearing your thoughts on the latest advancements in AI. You are optomistic. \
You are particularly interested in new AI methods and techniques, and highlighting interesting \
use cases of your LLM frameworks, LangChain LangGraph and LangSmith.

I will give you an article. Extract a bulleted list of key points made or statements made in it.

If there is not enough information to generate a good list of info, then call the `NotEnoughInfo` tool.
Otherwise, ONLY generate a bulleted list of facts Using the `Facts` tool.

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
                "content": convert_to_facts_system_prompt.format(article=content)
            },
        ]
    )

    if len(response.tool_calls) == 0:
        return {"facts": facts}
    else:
        tool_call = response.tool_calls[0]
        if tool_call['name'] == "Facts":
            return {"facts": tool_call['args']['facts']}

def write_tweet(state, config):
    print("tweets state: ", state)
    facts = state.get("facts")
    if not facts: 
        raise ValueError("No facts provided")
    
    tweet = ChatOpenAI().invoke(
        [
            {
                "role": "system", 
                "content": """
                You are Harrison Chase. You tweet highlighting information related to LangChain, your LLM company.
                You use emojis. You use exclamation points but are not overly enthusiastic.
             """
            },
            {
                "role": "human", 
                "content": "Write a tweet summarizing the following facts. Less than 20 words:  " + "\n".join(facts)
            }
        ]
    ).content

    attempts = state.get("attempts", 0) or 0

    return {"tweet": tweet, "attempts": attempts + 1}

# TODO: REPLACE WITH FEW SHOT SEARCH
def get_harrision_tweets_examples():
    return  """
        <example>
        It's out! LangChain v0.1.0 comes out with an improved package architecture for stability and production readiness, as well a focus on:
        👀 Observability
        ↔️ Integrations
        🔗 Composability
        🏳️ Streaming
        🧱 Output Parsing
        🔍 Retrieval
        🤖 Agents
        </example>

        <example>
        this is a really cool project - its agent that writes other agents
        </example>

        <example>
        This was a nights and weekend project for me, but I had a lot of fun making it and think there's some good opportunities to improve it
        See a walkthrough here: youtu.be/OM6ibrjn_Sg
        </example>

        <example>
        🪖 LangGraph Engineer
        This is an alpha version of an agent that can help bootstrap LangGraph applications
        It will focus on creating the correct nodes and edges, but will not attempt to write the logic to fill in the nodes and edges - rather will leave that for you
        Try out the deployed version: smith.langchain.com/studio/thread?…
        The agent consists of a few steps:

        1. Converse with the user to gather all requirements
        2. Write a draft
        3. Run programmatic checks against the generated draft (right now just checking that the response has the right format). If it fails, then go back to step 2. If it passes, then continue to step 4.
        4. Run an LLM critique against the generated draft. If it fails, go back to step 2. If it passes, the continue to the end.
        
        Deployed on LangGraph Cloud, and made publicly accessible (the ability to do this is behind a feature flag, DM me if interested)
        Code: github.com/hwchase17/lang…
        </example>

        <example>
        Opening up access for LangGraph Cloud!
        </example>

        <example>
        Once of the best things about LangGraph is the built in persistence layer
        This enables all sorts of human-in-the-loop interactions
        We've released LangGraph 0.2 which improves management of that and open-sourced our Postgres implementation
        </example>
    """



def harrison_critique(state, config):
    print("sounds like state: ", state)

    tweet = state.get("tweet")
    if not tweet:
        raise ValueError("No tweet provided")

    sounds_like_harrison = ChatOpenAI().with_structured_output(SoundsLikeHarrison).invoke(
        [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant. Here are past examples of tweets from Harrison Chase:
                    {get_harrision_tweets_examples()}
                 """
            },
            {
                "role": "human",
                "content": "Does the following tweet sound like Harrison Chase: " + tweet
            }
        ]
    )

    return {"critique": sounds_like_harrison}
def sounds_like_harrison_checker(state, config):
    attempts = state.get("attempts", 0)

    if attempts > 4:
        return "done"
    sounds_like_harrison = state['critique'].sounds_like_harrison
    if not sounds_like_harrison:
        return "revise"
    else:
        return "done"


reviser_human_prompt = """Revise the following tweet to be more in the style of Harrison Chase's past tweets.

Here is some direct feedback: {feedback}

Here's the tweet:

{tweet}"""

def harrison_tweet_reviser(state, config):
    print("reviser state: ", state)
    tweet = state.get("tweet")
    if not tweet: 
        raise ValueError("No tweet provided")
    
    revised_tweet = ChatOpenAI().invoke(
        [
            {
                "role": "system", 
                "content": f"""
                You are a helpful assistant. 

                Here are past examples of tweets from Harrison Chase:
                {get_harrision_tweets_examples()}
             """
            },
            {
                "role": "human", 
                "content": reviser_human_prompt.format(tweet=tweet, feedback=state['critique'].critique)
            }
        ]
    ).content

    curr_attempts = state.get("attempts", 0)
    print(f"curr_attempts: {curr_attempts}")
    return {"tweet": revised_tweet, "attempts": curr_attempts + 1}


def _check_if_enough_info(state: DirectSummarizerState) -> Literal["write_tweet", END]:
    if "facts" in state:
        return "write_tweet"
    else:
        return END


summarizer_workflow = StateGraph(DirectSummarizerState, input=GraphInput, output=GraphOutput)
summarizer_workflow.add_node(get_contents)
summarizer_workflow.add_node(convert_to_facts)
summarizer_workflow.add_node(harrison_critique)
summarizer_workflow.add_node(write_tweet)
summarizer_workflow.add_node(harrison_tweet_reviser)

summarizer_workflow.set_entry_point("get_contents")
summarizer_workflow.add_edge("get_contents", "convert_to_facts")
summarizer_workflow.add_edge("write_tweet", "harrison_critique")
summarizer_workflow.add_edge("harrison_tweet_reviser", "harrison_critique")
summarizer_workflow.add_conditional_edges("convert_to_facts", _check_if_enough_info)


summarizer_workflow.add_conditional_edges(
    "harrison_critique",
    sounds_like_harrison_checker, 
    {
        "revise": "harrison_tweet_reviser",
        "done": END
    }
)


summarizer_graph = summarizer_workflow.compile()

