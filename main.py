from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

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
    nn.Sigmoid()  # For multi-label classification
)
mobilenet.eval()

# Define input schema
class ImageInput(BaseModel):
    image_base64: str

@app.post("/classify")
def classify_image(input: ImageInput):
    try:
        # Decode base64 image
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess and predict
        input_tensor = processor(image).unsqueeze(0)
        with torch.no_grad():
            outputs = mobilenet(input_tensor)
            probs = outputs[0].cpu().numpy()

        # Use a threshold to select active labels
        threshold = 0.3
        results = [
            {"label": LABELS[i], "confidence": round(float(prob), 3)}
            for i, prob in enumerate(probs)
            if prob > threshold
        ]

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}

@app.post("/wastechart")
def generate_pie_chart(input: ImageInput):
    try:
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        input_tensor = processor(image).unsqueeze(0)
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
            <h3>Waste Composition (Multi-label)</h3>
            <canvas id="wasteChart" width="400" height="400"></canvas>
            <script>
                const ctx = document.getElementById('wasteChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'pie',
                    data: {{
                        labels: {labels_js},
                        datasets: [{{
                            label: 'Waste Composition',
                            data: {data_js},
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.6)',
                                'rgba(54, 162, 235, 0.6)',
                                'rgba(255, 206, 86, 0.6)',
                                'rgba(75, 192, 192, 0.6)',
                                'rgba(153, 102, 255, 0.6)',
                                'rgba(255, 159, 64, 0.6)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)',
                                'rgba(255, 159, 64, 1)'
                            ],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return {"html": html}

    except Exception as e:
        return {"error": str(e)}
