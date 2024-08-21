import re
from pytube import YouTube
from langchain_community.document_loaders import WebBaseLoader
import requests

import bs4


def is_youtube_url(url):
    # Regular expression pattern for YouTube URLs
    youtube_regex = (
        r"(https?://)?(www\.)?"
        "(youtube|youtu|youtube-nocookie)\.(com|be)/"
        "(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )

    youtube_regex_match = re.match(youtube_regex, url)
    return bool(youtube_regex_match)


def is_medium_url(url):
    # Regular expression pattern for Medium URLs
    medium_regex = r"https?://medium\.com/"
    medium_regex_match = re.match(medium_regex, url)
    return bool(medium_regex_match)


def is_github_url(url):
    match = re.match(r"https?://github.com/([^/]+)/([^/]+)", url)
    return bool(match)


def get_github_readme(url):
    # Extract owner and repo from the GitHub URL
    match = re.match(r"https?://github.com/([^/]+)/([^/]+)", url)

    owner, repo = match.groups()
    readme_files = ["README.md", "README.txt", "README", "Readme.md", "readme.md"]

    for branch in ["main", "master"]:
        for filename in readme_files:
            # Construct the raw content URL
            raw_url = (
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
            )

            try:
                response = requests.get(raw_url)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                return response.text
            except requests.exceptions.HTTPError as e:
                print("ERROR")
                print(e)
                continue

    print("Could not easily find a README file")
    return ""


def get_youtube_description(url):
    # Create a YouTube object
    yt = YouTube(url)

    return f"Title: {yt.title}\n\nDescription: {yt.description}"


def get_medium_content(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer("article")
        ),  # only elements in article tag
    )

    docs = loader.load()
    return docs[0].page_content


def get_content(url):
    print("get_content url: ", url)
    if is_youtube_url(url):
        return get_youtube_description(url)
    elif is_medium_url(url):
        return get_medium_content(url)
    elif is_github_url(url):
        return get_github_readme(url)
    else:
        print("generic url")
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
