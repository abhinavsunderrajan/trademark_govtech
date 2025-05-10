import requests
import base64
from PIL import Image
from io import BytesIO
import time

# Load and encode an image
image_path = "./images/90943a27-389d-46b0-833d-f9fa6b2bc10f.jpg"
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Make the API request
start_time = time.time()
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "image_base64": encoded_string,
        "instruction": "Analyse the image carefully and return the text in English, Chinese and the description of any pattern if present. Return the answer as json string."
    }
)

# Print the result
print(response.json())
end_time = time.time()
print(f"Time taken for response: {end_time - start_time} seconds")