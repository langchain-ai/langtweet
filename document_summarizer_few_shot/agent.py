from typing import TypedDict

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

class DirectSummarizerState(TypedDict):
    content: str
    facts: list[str]
    tweet: str
    attempts: int

def convert_to_facts(state, config):
    print("facts state: ", state)
    content = state.get("content")
    if not content: 
        raise ValueError("No content provided")
    
    facts = ChatOpenAI().invoke(
        [
            {
                "role": "system", 
                "content": """
                You are Harrison Chase. You run a company focused on the latest AI technologies. Your
                followers love hearing your thoughts on the latest advancements in AI. You are optomistic.
                You are particularly interested in new AI methods and techniques, and highlighting interesting
                use cases of your LLM framework, LangChain.
             """
            },
            {
                "role": "human", 
                "content": "Extract a bulleted list of key points made or facts stated in the following document:  " + content
            }
        ]
    ).content

    return {"facts": facts}

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


class SoundsLikeHarrison(BaseModel):
    sounds_like_harrison: bool = Field(description="whether the tweet sounds like harrison chase")

def sounds_like_harrison_checker(state, config):
    print("sounds like state: ", state)
    attempts = state.get("attempts", 0)
    if attempts > 4:
        return "done"
    
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
    ).sounds_like_harrison

    if not sounds_like_harrison:
        return "revise"
    else:
        return "done"


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
                "content": "Revise the following tweet to be more in the style of Harrison Chase's past tweets: " + tweet
            }
        ]
    ).content

    curr_attempts = state.get("attempts", 0)
    print(f"curr_attempts: {curr_attempts}")
    return {"tweet": revised_tweet, "attempts": curr_attempts + 1}




summarizer_workflow = StateGraph(DirectSummarizerState)

summarizer_workflow.add_node("convert_to_facts", convert_to_facts)
summarizer_workflow.set_entry_point("convert_to_facts")

summarizer_workflow.add_node("write_tweet", write_tweet)
summarizer_workflow.add_edge("convert_to_facts", "write_tweet")

summarizer_workflow.add_node("harrison_tweet_reviser", harrison_tweet_reviser)

summarizer_workflow.add_conditional_edges(
    "write_tweet",
    sounds_like_harrison_checker, 
    {
        "revise": "harrison_tweet_reviser",
        "done": END
    }
)

summarizer_workflow.add_conditional_edges(
    "harrison_tweet_reviser",
    sounds_like_harrison_checker, 
    {
        "revise": "harrison_tweet_reviser",
        "done": END
    }
)

summarizer_graph = summarizer_workflow.compile()

