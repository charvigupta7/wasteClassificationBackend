from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
import io
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("watersplash/waste-classification")
model = AutoModelForImageClassification.from_pretrained("watersplash/waste-classification")

class ImageInput(BaseModel):
    image_base64: str

@app.post("/classify")
async def classify_image(payload: ImageInput):
    try:
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]

            probs = F.softmax(logits, dim=-1)
            confidence = probs[0][predicted_class_idx].item()

        return {
            "label": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
