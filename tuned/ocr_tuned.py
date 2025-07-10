from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://dev-gpu.roboflow.cloud",
    api_key=os.getenv("API_KEY")
)

result = client.run_workflow(
    workspace_name="dev-m9yee",
    workflow_id="vlm-ocr-finetune",
    images={
        "image": "sample/test.png"
    },
    use_cache=True # cache workflow definition for 15 minutes
)

# for num in result[0]["smol_vlm"]:
