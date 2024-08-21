# LangTweet

This is an example repository demonstrating the power of dynamic few-shot prompting.
We use LangSmith to serve a few-shot dataset that we use to prompt an LLM to generate tweets in a style we like.
We use LangGraph to orchestrate the (relatively simple) graph.

**Key Links:**
- [YouTube Walkthrough]()
- [Try out the graph here]()
- [LangSmith]()
- [LangGraph]()

## The graph

The graph logic is very simple for now.

First, we load the content of a given url.
You can find the logic for loading the content in `langtweet/loading.py`.

After that, we pass the content to a prompt.
This prompt contains some basic instructions, but more importantly a few examples of similar tweets in the past.
The logic for this can be found in `langtweet/agent.py`

## The dynamic few-shot selection

A key part of this application is using dynamic few-shot selection to help prompt.
The prompt instructions for tweeting are pretty basic.
It provides a bit of context then tells the LLM to pay attention to the examples.
Therefor, the examples are doing a lot of lifting here.

We use LangSmith to manage and serve the dataset that we use as examples.

## The deployment

We deploy the graph to LangGraph Cloud.
This gives us a nice API to interact with, as well as fun studio UI to use to try out the graph.

We've made this studio publicly accessible, you can use it [here](TODO).

## The feedback loop

A key part of a system like this is a feedback loop to continue to gather examples that can be used in the future.
To that end, we have provided a very simple Streamlit application for invoking the graph and then correcting it and giving feedback.
This feedback automatically creates a new entry in the dataset, which can then be used in the future.
