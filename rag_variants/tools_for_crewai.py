import json
import os

from exa_py import Exa
from langchain.agents import tool
from matplotlib.font_manager import json_load


class JsonDocTool:
    def __init__(self, json_fn: str) -> None:
        self.json_fn = json_fn

    @tool
    def convert_to_json(self):
        """return a JSON payload after reading a file."""
        payload = json_load(self.json_fn)
        return json.dumps(payload)


class ExaSearchTool:
    @tool
    def search(query: str):
        """Search for a webpage based on the query."""
        return ExaSearchTool._exa().search(
            f"{query}", use_autoprompt=True, num_results=3
        )

    @tool
    def find_similar(url: str):
        """Search for webpages similar to a given URL.
        The url passed in should be a URL returned from `search`.
        """
        return ExaSearchTool._exa().find_similar(url, num_results=3)

    @tool
    def get_contents(ids: str):
        """Get the contents of a webpage.
        The ids must be passed in as a list, a list of ids returned from `search`.
        """
        ids = eval(ids)
        contents = str(ExaSearchTool._exa().get_contents(ids))
        print(contents)
        contents = contents.split("URL:")
        contents = [content[:1000] for content in contents]
        return "\n\n".join(contents)

    def tools():
        return [
            ExaSearchTool.search,
            ExaSearchTool.find_similar,
            ExaSearchTool.get_contents,
        ]

    def _exa():
        return Exa(api_key=os.environ["EXA_API_KEY"])
