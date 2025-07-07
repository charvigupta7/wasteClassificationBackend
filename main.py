from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
import base64

app = FastAPI()

# Load model and processor only once
processor = AutoImageProcessor.from_pretrained("watersplash/waste-classification")
model = AutoModelForImageClassification.from_pretrained("watersplash/waste-classification")

# Input schema for API
class ImageInput(BaseModel):
    image_base64: str  # base64-encoded image from client

@app.post("/classify")
async def classify_image(payload: ImageInput):
    try:
        # Decode base64 image
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]

        return {"label": label}

    except Exception as e:
        return {"error": str(e)}
