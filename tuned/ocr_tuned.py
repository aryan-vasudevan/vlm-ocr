from inference_sdk import InferenceHTTPClient
import os
import re
from dotenv import load_dotenv
import cv2
import time

# Start timer
start_time = time.time()

# Load environment variables
load_dotenv()

# Initialize the inference client
client = InferenceHTTPClient(
    api_url=os.getenv("API_URL"),
    api_key=os.getenv("API_KEY")
)

player_set = set()

# Load video and determine frames
video_path = "sample/clip.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames = 2
step = max(frame_count // total_frames, 1)
frame_indices = [i * step for i in range(total_frames)]

current_frame = 0
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if current_frame in frame_indices:
        # Save frame as jpg
        frame_path = f"dump/frame_{frame_id}_{time.time()}.jpg"
        cv2.imwrite(frame_path, frame)

        # Run inference using the saved file
        result = client.run_workflow(
            workspace_name="dev-m9yee",
            workflow_id="vlm-ocr-finetune",
            images={"image": frame_path},
            use_cache=True
        )
        print(f"{frame_id + 1}/{total_frames}")

        # Extract player numbers
        for res in result[0]["smol_vlm"]:
            player_set.add(res)

        frame_id += 1

    current_frame += 1

cap.release()

# End timer
end_time = time.time()
duration = end_time - start_time

print("Unique Player Numbers:", player_set)
print(f"Total Time Elapsed: {duration:.2f} seconds")
