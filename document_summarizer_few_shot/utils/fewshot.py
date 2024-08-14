import json
from typing import Any, Dict, List

from langchain_core.example_selectors import BaseExampleSelector
from langsmith import client, utils as ls_utils, schemas as ls_schemas, traceable

langsmith_client = client.Client()

@traceable
def search_similar_examples(
    dataset_id: str, 
    inputs_dict: dict[str, Any], 
    limit: int = 5,
    debug: bool = True) -> list[dict[str, Any]]:
    search_req_json = json.dumps(
        {
            "inputs": inputs_dict,
            "limit": limit,
            "debug": debug
        }
    )
    few_shot_resp = langsmith_client.request_with_retries(
        "POST",
        f"/datasets/{dataset_id}/search",
        headers={**langsmith_client._headers, "Content-Type": "application/json"},
        data=search_req_json
    )
    ls_utils.raise_for_status_with_text(few_shot_resp)
    return few_shot_resp.json()


class LangSmithDatasetExampleSelector(BaseExampleSelector):

    def __init__(
        self,
        dataset_id,
        num_examples: int = 5):
        self.dataset_id = dataset_id
        self.num_examples = num_examples

    def add_example(self, example: Dict[str, str]) -> Any:
        raise NotImplementedError()

    @traceable
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        return search_similar_examples(
            dataset_id=self.dataset_id,
            inputs_dict=input_variables,
            limit=self.num_examples
        )["examples"]