from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login
import time
import gc
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and login to HuggingFace
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

app = FastAPI(title="Qwen2-VL Image Analysis API")

# Initialize model and processor with memory optimizations
def load_model():
    try:
        # Set memory optimization flags
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log available memory
        process = psutil.Process()
        logger.info(f"Memory usage before model load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )

        base_model.load_adapter("final_chkpt", device_map="auto")
        
        # Log memory after model load
        logger.info(f"Memory usage after model load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        print(base_model.device)
        return base_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_processor():
    return AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Initialize model and processor
model = load_model()
processor = load_processor()

class ImageRequest(BaseModel):
    image_base64: str
    instruction: str = """
    Analyse the image carfully and return the text in English, Chinese and the description of any pattern if present. Return the answer as json string.
    """

def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        # Remove the data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data), quality=70)
        
        # Resize image if too large to save memory
        if image.width > 512 or image.height > 512:
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        if image.width<30 or image.height<30:
            image = image.resize((max(image.width, 30), max(image.height, 30)), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/ping")
def pong():
    return {"ping": "pong!"}

@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    try:
        # Clear memory before processing
        torch.cuda.empty_cache()
        gc.collect()
        
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Prepare message
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.instruction},
            ],
        }]
        
        # Process the image
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device and generate
        inputs = inputs.to(model.device)
        start_time = time.time()
        
        with torch.inference_mode():
            op = model.generate(**inputs, max_new_tokens=500, use_cache=True, temperature=0.1)
        
        # Process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, op)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clear memory after processing
        del inputs
        del op
        del generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "result": output_text[0],
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 