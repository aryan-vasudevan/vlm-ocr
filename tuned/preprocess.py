from inference import InferencePipeline
import os
from dotenv import load_dotenv

load_dotenv()

def added():
    print("added")

# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=os.getenv("API_KEY"),
    workspace_name="dev-m9yee",
    workflow_id="vlm-ocr-preprocess",
    video_reference="../sample/clip2.mp4",
    on_prediction=added,
    max_fps=10,
)

pipeline.start()
pipeline.join()
