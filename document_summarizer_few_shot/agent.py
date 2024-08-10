from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from document_summarizer_few_shot.utils.nodes import call_model, should_continue, tool_node
from document_summarizer_few_shot.utils.state import AgentState, DocSummarizerState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()




summarizer_workflow = StateGraph(DocSummarizerState, config_schema=GraphConfig)

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.github import GithubFileLoader

from langchain_openai import ChatOpenAI

# def load_url_generic(state, config):
#     url = state.get("url")
#     if not url: 
#         raise ValueError("No URL provided")
    
#     loader = WebBaseLoader(url)
#     doc = loader.load()[0].page_content
    

#     return {"url": url, "raw_content": doc}

def convert_to_repo_name(state, config):
    url = state.get("url")
    if not url: 
        raise ValueError("No URL provided")
    
    repo_name = ChatOpenAI().invoke(
        [
            {
                "role": "human", 
                "content": """
                You convert URLs into the github repo name. The repo name is of the form <owner>/<repo_name>
                after the github.com/ part of the URL.

                <example>
                    <url>https://github.com/langchain-ai/langchainplus/tree/main/smith-backend</url>
                    <output>langchain-ai/langchainplus</output>
                </example>

                <example>
                    <url>https://github.com/apache/spark/blob/master/README.md</url>
                    <output>apache/spark</output>
                </example>
             """
            },
            {
                "role": "human", 
                "content": "What is the github repo name? Answer in the form <owner>/<repo_name>. Repo URL: " + url
            }
        ]
    ).content

    return {"repo_name": repo_name}

def load_url_github(state, config):
    repo_name = state.get("repo_name")
    if not repo_name: 
        raise ValueError("No repo_name provided")
    
    print("repo_name", repo_name)
    loader = GithubFileLoader(
        repo=repo_name,  # the repo name
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(
            ".md"
        ),  # load all markdowns files.
        branch="master"
    )
    doc = loader.load()
    

    return {"documents": doc}


def convert_to_facts(state, config):
    docs = state.get("documents")
    if not docs: 
        raise ValueError("No documents provided")
    
    formatted_documents = "\n".join(
        [
            f"""
            <document>
                <content>{doc.page_content}</content>
            </document>
            """
            for doc in docs
        ]
    )
    
    facts = ChatOpenAI().invoke(
        [
            {
                "role": "system", 
                "content": """
                You are a helpful assistant. 
             """
            },
            {
                "role": "human", 
                "content": "Extract a bulleted list of facts from the following documents:  " + formatted_documents
            }
        ]
    ).content

    return {"facts": facts}

# summarizer_workflow.add_node("load_url", load_url_generic)
summarizer_workflow.add_node("convert_to_repo_name", convert_to_repo_name)
summarizer_workflow.set_entry_point("convert_to_repo_name")

summarizer_workflow.add_node("load_url_github", load_url_github)
summarizer_workflow.add_edge("convert_to_repo_name", "load_url_github")

summarizer_workflow.add_node("convert_to_facts", convert_to_facts)
summarizer_workflow.add_edge("load_url_github", "convert_to_facts")


summarizer_graph = summarizer_workflow.compile()


class DirectSummarizerState(TypedDict):
    content: str
    facts: list[str]
    tweet: str


def convert_to_facts_v2(state, config):
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
    facts = state.get("facts")
    if not facts: 
        raise ValueError("No facts provided")
    
    tweet = ChatOpenAI().invoke(
        [
            {
                "role": "system", 
                "content": """
                You are Harrison Chase. You run a company focused on the latest AI technologies. Your
                followers love hearing your thoughts on AI. You are optomistic.
                You are particularly interested in new AI methods and techniques, and highlighting interesting
                use cases of your LLM framework, LangChain.
             """
            },
            {
                "role": "human", 
                "content": "Write a tweet summarizing the following facts. Less than 20 words:  " + "\n".join(facts)
            }
        ]
    ).content

    return {"tweet": tweet}


summarizer_workflow_v2 = StateGraph(DirectSummarizerState, config_schema=GraphConfig)

summarizer_workflow_v2.add_node("convert_to_facts", convert_to_facts_v2)
summarizer_workflow_v2.set_entry_point("convert_to_facts")

summarizer_workflow_v2.add_node("write_tweet", write_tweet)
summarizer_workflow_v2.add_edge("convert_to_facts", "write_tweet")

summarizer_graph_v2 = summarizer_workflow_v2.compile()

