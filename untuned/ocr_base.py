from inference_sdk import InferenceHTTPClient
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = InferenceHTTPClient(
    api_url=os.getenv("API_URL"),
    api_key=os.getenv("API_KEY")
)

result = client.run_workflow(
    workspace_name="dev-m9yee",
    workflow_id="vlm-ocr-base",
    images={
        "image": "sample/test.png"
    },
    use_cache=True
)

player_set = set()

pattern = re.compile(r"Assistant:\s*([0-9]+)")
for res in result[0]["smol_vlm"]:
    m = pattern.search(res)
    if m:
        number = m.group(1)
        player_set.add(number)

print(player_set)