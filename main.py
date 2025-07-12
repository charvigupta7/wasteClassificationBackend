from fastapi import FastAPI
from pydantic import BaseModel, model_validator
import requests
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import List, Dict, Union, Optional
from collections import defaultdict

app = FastAPI()

# --- Step 1: Define waste category labels for MobileNet
LABELS = ["plastic", "metal", "glass", "paper", "organic", "e-waste"]

# --- Step 2: Preprocessing pipeline
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Step 3: Load and modify MobileNetV2 for multi-label classification
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(mobilenet.last_channel, len(LABELS)),
    nn.Sigmoid()
)
mobilenet.eval()

# --- Step 4: Input schema
class ImageInput(BaseModel):
    predictions: Optional[List[Dict[str, Union[str, float]]]] = None
    image_url: Optional[str] = None

    @model_validator(mode="after")
    def check_one_required(cls, values):
        if not values.predictions and not values.image_url:
            raise ValueError("Either 'predictions' or 'image_url' must be provided.")
        return values

# --- Step 5: Image loader
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return processor(image).unsqueeze(0)

# --- Step 6: Multi-label classification route
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

# --- Step 7: Waste composition using Roboflow YOLOv8
def get_waste_composition_from_roboflow(image_url: str, api_key: str):
    endpoint = f"https://detect.roboflow.com/taco-trash-annotations-in-context/16?api_key={api_key}"

    # Download image bytes from the given URL
    response = requests.get(image_url)
    image_bytes = response.content

    # Send image to Roboflow's model endpoint
    upload_response = requests.post(
        endpoint,
        files={"file": image_bytes},
        data={"confidence": "0.25", "overlap": "0.2"}
    )
    result = upload_response.json()

    # Process bounding box results
    boxes = result.get("predictions", [])
    areas_by_class = defaultdict(float)

    for pred in boxes:
        width = pred["width"]
        height = pred["height"]
        area = width * height
        label = pred["class"]
        areas_by_class[label] += area

    total_area = sum(areas_by_class.values())
    if total_area == 0:
        return {"labels": [], "composition": [], "total_area": 0}

    composition = [
        {
            "label": label,
            "percentage": round((area / total_area) * 100, 2),
            "area": round(area, 2)
        }
        for label, area in areas_by_class.items()
    ]

    return {
        "labels": list(areas_by_class.keys()),
        "composition": sorted(composition, key=lambda x: -x["percentage"]),
        "total_area": round(total_area, 2)
    }

# --- Step 8: API route to get waste composition from Roboflow
@app.post("/wastecomposition")
def waste_composition_api(input: ImageInput):
    try:
        return get_waste_composition_from_roboflow(
            image_url=input.image_url,
            api_key="YOUR_ROBOFLOW_API_KEY"  # Replace this with your actual API key
        )
    except Exception as e:
        return {"error": str(e)}
