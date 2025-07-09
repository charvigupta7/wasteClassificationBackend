from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import torch
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification

app = FastAPI()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("akmalia31/trash-classification-cnn-mobilnetv2")
model = MobileNetV2ForImageClassification.from_pretrained("akmalia31/trash-classification-cnn-mobilnetv2")

class ImageInput(BaseModel):
    image_base64: str

@app.post("/classify")
def classify_image(input: ImageInput):
    try:
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Process and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        topk = torch.topk(probs, k=3)  # Top 3 predictions
        labels = model.config.id2label

        results = [
            {"label": labels[idx.item()], "confidence": round(score.item(), 3)}
            for idx, score in zip(topk.indices, topk.values)
        ]

        return {"top_predictions": results}

    except Exception as e:
        return {"error": str(e)}

@app.post("/wastechart")
def generate_pie_chart(input: ImageInput):
    try:
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run classification
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label

        # Get top 5 predictions for pie chart
        topk = torch.topk(probs, k=5)
        filtered = [
            {"label": labels[i.item()], "confidence": round(p.item(), 3)}
            for i, p in zip(topk.indices, topk.values)
            if p.item() > 0.01
        ]

        if not filtered:
            return {"html": "<p>No waste types detected with sufficient confidence.</p>"}

        total = sum(item["confidence"] for item in filtered)
        composition = [
            {
                "label": item["label"],
                "percentage": round((item["confidence"] / total) * 100, 2)
            }
            for item in filtered
        ]

        labels_js = [item["label"] for item in composition]
        data_js = [item["percentage"] for item in composition]

        html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h3>Waste Composition (Top Predictions)</h3>
