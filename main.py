from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
import io
import base64
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer, AutoModelForCausalLM, pipeline


app = FastAPI()

# Load model and processor at startup
llm_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
llm_model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto")
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

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
        
        prompt = f"The image contains waste identified as '{label}' with {confidence*100:.2f}% confidence. Write a concise summary of this classification."
        llm_response = llm_pipe(prompt)[0]['generated_text']

        return {
            "summary": llm_response.strip(),
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------- /recommendation -------------------
@app.post("/recommendation")
async def generate_recommendation(payload: ImageInput):
    try:
        label, confidence = classify_base64_image(payload.image_base64)
        prompt = f"The waste is classified as '{label}' with {confidence*100:.2f}% confidence. What type of waste is it (biodegradable or non-biodegradable), and how should it be disposed of?"

        llm_response = llm_pipe(prompt)[0]['generated_text']

        return {
            "label": label,
            "confidence": confidence,
            "recommendation": llm_response.strip()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
