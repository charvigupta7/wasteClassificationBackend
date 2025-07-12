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
import os
from dotenv import load_dotenv

load_dotenv() 
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

EXCLUDED_LABELS = {"mixed", "trash", "mixed trash"}

app = FastAPI()

class ImageInput(BaseModel):
    predictions: Optional[List[Dict[str, Union[str, float]]]] = None
    image_url: Optional[str] = None

    @model_validator(mode="after")
    def check_one_required(cls, values):
        if not values.predictions and not values.image_url:
            raise ValueError("Either 'predictions' or 'image_url' must be provided.")
        return values

# --- Roboflow waste composition function
def infer_roboflow(image_url: str, api_key: str):
    endpoint = f"https://detect.roboflow.com/taco-trash-annotations-in-context/16?api_key={api_key}"

    response = requests.get(image_url)
    image_bytes = response.content

    upload_response = requests.post(
        endpoint,
        files={"file": image_bytes},
        data={"confidence": "0.25", "overlap": "0.2"}
    )
    return upload_response.json()

# --- Classify waste types (just labels, no area calc)
@app.post("/classify")
def classify_waste_types(input: ImageInput):
    try:
        result = infer_roboflow(input.image_url, ROBOFLOW_API_KEY)
        boxes = result.get("predictions", [])

        # Collect unique labels
        labels = sorted(set(pred["class"] for pred in boxes))

        return {"labels": labels}

    except Exception as e:
        return {"error": str(e)}

# --- Waste composition (as before)
@app.post("/wastecomposition")
def waste_composition_api(input: ImageInput):
    try:
        result = infer_roboflow(input.image_url, ROBOFLOW_API_KEY)
        boxes = result.get("predictions", [])

        areas_by_class = defaultdict(float)
        for pred in boxes:
            area = pred["width"] * pred["height"]
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
            if label.lower() not in EXCLUDED_LABELS
        ]

        return {
            "labels": list(areas_by_class.keys()),
            "composition": sorted(composition, key=lambda x: -x["percentage"]),
            "total_area": round(total_area, 2)
        }

    except Exception as e:
        return {"error": str(e)}
