from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
import torch.nn.functional as F

app = FastAPI()

# Load model and processor once
processor = AutoImageProcessor.from_pretrained("watersplash/waste-classification")
model = AutoModelForImageClassification.from_pretrained("watersplash/waste-classification")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read file content into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]

            # Optional: confidence score
            probs = F.softmax(logits, dim=-1)
            confidence = probs[0][predicted_class_idx].item()

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "filename": file.filename
        }

    except Exception as e:
        return {"error": "Failed to classify image", "details": str(e)}
