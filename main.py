from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
import io
import base64
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI()

# Load model and processor at startup
processor = AutoImageProcessor.from_pretrained("watersplash/waste-classification")
model = AutoModelForImageClassification.from_pretrained("watersplash/waste-classification")

# ------------------- Input Schema -------------------
class ImageInput(BaseModel):
    image_base64: str

# ------------------- Waste Knowledge Base -------------------
WASTE_KNOWLEDGE = {
    "Plastic": {"type": "Non-Biodegradable", "action": "Recycle at designated centers or reuse."},
    "Food waste": {"type": "Biodegradable", "action": "Compost or dispose in green bin."},
    "Paper": {"type": "Biodegradable", "action": "Recycle or compost if not contaminated."},
    "Metal": {"type": "Non-Biodegradable", "action": "Recycle at authorized metal scrap vendors."},
    "E-waste": {"type": "Non-Biodegradable", "action": "Dispose at e-waste collection centers."},
    "Glass": {"type": "Non-Biodegradable", "action": "Recycle if not broken; else, dispose safely."},
    "Cardboard": {"type": "Biodegradable", "action": "Recycle or compost."}
}

# ------------------- Classification Utility -------------------
def classify_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        probs = F.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()
    return label, round(confidence, 4)

# ------------------- /classify -------------------
@app.post("/classify")
async def classify_image(payload: ImageInput):
    try:
        label, confidence = classify_base64_image(payload.image_base64)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------- /summary -------------------
@app.post("/summary")
async def generate_summary(payload: ImageInput):
    try:
        label, confidence = classify_base64_image(payload.image_base64)
        summary = f"The uploaded waste item is classified as '{label}' with {confidence*100:.2f}% confidence."
        return {"summary": summary, "label": label, "confidence": confidence}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------- /recommendation -------------------
@app.post("/recommendation")
async def generate_recommendation(payload: ImageInput):
    try:
        label, confidence = classify_base64_image(payload.image_base64)
        normalized_label = label.strip().title()  # e.g., "plastic" -> "Plastic"
        info = WASTE_KNOWLEDGE.get(normalized_label, {
            "type": "Unknown",
            "action": "No specific advice. Refer to local waste guidelines."
        })
        
        return {
            "label": label,
            "confidence": confidence,
            "waste_type": info["type"],
            "recommended_action": info["action"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
