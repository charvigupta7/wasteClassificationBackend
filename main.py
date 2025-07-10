from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import requests
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI()

LABELS = ["plastic", "metal", "glass", "paper", "organic", "e-waste"]

processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(mobilenet.last_channel, len(LABELS)),
    nn.Sigmoid()
)
mobilenet.eval()


# ==== Input Schemas ====

class ImageInput(BaseModel):
    image_url: str

class Prediction(BaseModel):
    label: str
    confidence: float

class WasteCompositionInput(BaseModel):
    image_url: Optional[str] = None  # Optional, just for traceability
    predictions: List[Prediction]


# ==== Utility ====

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return processor(image).unsqueeze(0)


# ==== Endpoints ====

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
def generate_waste_composition(input: WasteCompositionInput):
    try:
        filtered = [
            {"label": p.label, "confidence": round(float(p.confidence), 3)}
            for p in input.predictions
            if p.confidence > 0.1
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

# @app.post("/wastechart")
# def generate_pie_chart(input: ImageInput):
#     try:
#         input_tensor = load_image_from_url(input.image_url)
#         with torch.no_grad():
#             outputs = mobilenet(input_tensor)
#             probs = outputs[0].cpu().numpy()

#         threshold = 0.1
#         filtered = [
#             {"label": LABELS[i], "confidence": round(float(p), 3)}
#             for i, p in enumerate(probs)
#             if p > threshold
#         ]

#         if not filtered:
#             return {"html": "<p>No waste types detected with sufficient confidence.</p>"}

#         total = sum(item["confidence"] for item in filtered)
#         composition = [
#             {
#                 "label": item["label"],
#                 "percentage": round((item["confidence"] / total) * 100, 2)
#             }
#             for item in filtered
#         ]

#         labels_js = json.dumps([item["label"] for item in composition])
#         data_js = json.dumps([item["percentage"] for item in composition])

#         html = f"""
# <!DOCTYPE html>
# <html>
# <head>
#   <title>Waste Composition Pie Chart</title>
#   <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
# </head>
# <body>
#   <div style=\"max-width:400px; margin: 0 auto;\">
#     <h3>Waste Composition Pie Chart</h3>
#     <canvas id=\"wasteChart\" width=\"400\" height=\"400\"></canvas>
#   </div>
#   <script>
#     window.addEventListener('DOMContentLoaded', function() {{
#       const ctx = document.getElementById('wasteChart').getContext('2d');
#       new Chart(ctx, {{
#         type: 'pie',
#         data: {{
#           labels: {labels_js},
#           datasets: [{{
#             data: {data_js},
#             backgroundColor: [
#               '#4CAF50',
#               '#e0e0e0'
#             ],
#             borderColor: [
#               '#388E3C',
#               '#bdbdbd'
#             ],
#             borderWidth: 1
#           }}]
#         }},
#         options: {{
#           responsive: true,
#           plugins: {{
#             legend: {{
#               position: 'bottom'
#             }}
#           }}
#         }}
#       }});
#     }});
#   </script>
# </body>
# </html>
# """

#         return {"html": html}

#     except Exception as e:
#         return {"error": str(e)}
