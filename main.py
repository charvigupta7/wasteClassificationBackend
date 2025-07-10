from fastapi import FastAPI
from pydantic import BaseModel, model_validator
import requests
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
from typing import List, Dict, Union, Optional

app = FastAPI()

# Define waste category labels
LABELS = ["plastic", "metal", "glass", "paper", "organic", "e-waste"]

# Preprocessing pipeline
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load and modify MobileNetV2 for multi-label classification
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(mobilenet.last_channel, len(LABELS)),
    nn.Sigmoid()
)
mobilenet.eval()

# Define input schema
class ImageInput(BaseModel):
    predictions: Optional[List[Dict[str, Union[str, float]]]] = None
    image_url: Optional[str] = None

    @model_validator(mode="after")
    def check_one_required(cls, values):
        if not values.predictions and not values.image_url:
            raise ValueError("Either 'predictions' or 'image_url' must be provided.")
        return values

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return processor(image).unsqueeze(0)

@app.post("/classify")
def classify_image(input: ImageInput):
    try:
        input_tensor = load_image_from_url(input.image_url)
        with torch.no_grad():
            outputs = mobilenet(input_tensor)
            probs = outputs[0].cpu().numpy()

        threshold = 0.3
        results = [
            {"label": LABELS[i], "confidence": round(float(prob), 3)}
            for i, prob in enumerate(probs)
            if prob > threshold
        ]
        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}


@app.post("/wastecomposition")
def generate_waste_composition(input: ImageInput):
    try:
        input_tensor = load_image_from_url(input.image_url)
        with torch.no_grad():
            outputs = mobilenet(input_tensor)
            probs = outputs[0].cpu().numpy()

        threshold = 0.1
        filtered = [
            {"label": LABELS[i], "confidence": round(float(p), 3)}
            for i, p in enumerate(probs)
            if p > threshold
        ]

        if not filtered:
            return {
                "labels": [],
                "percentages": []
            }

        total = sum(item["confidence"] for item in filtered)
        composition = [
            {
                "label": item["label"],
                "percentage": round((item["confidence"] / total) * 100, 2)
            }
            for item in filtered
        ]

        labels = [item["label"] for item in composition]
        percentages = [item["percentage"] for item in composition]

        return {
            "labels": labels,
            "percentages": percentages
        }

    except Exception as e:
        return {"error": str(e)}

