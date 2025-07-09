from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, MobileNetV2ForImageClassification

app = FastAPI()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

class ImageInput(BaseModel):
    image_base64: str

@app.post("/multiclassify_composition")
def classify_and_estimate_composition(input: ImageInput):
    try:
        # Decode base64 image
        image_data = base64.b64decode(input.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get multi-label probabilities
        probs = torch.sigmoid(outputs.logits)[0]
        labels = model.config.id2label

        # Threshold and filter results
        raw_results = [
            {"label": labels[i], "confidence": p.item()}
            for i, p in enumerate(probs)
            if p.item() > 0.2  # Adjustable threshold
        ]

        # Normalize confidences to percentages for composition
        total_conf = sum(item["confidence"] for item in raw_results)
        if total_conf == 0:
            return {"composition": []}

        composition = [
            {
                "label": item["label"],
                "confidence": round(item["confidence"], 3),
                "percentage": round((item["confidence"] / total_conf) * 100, 1)
            }
            for item in raw_results
        ]

        return {"composition": composition}

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

        probs = torch.sigmoid(outputs.logits)[0]
        labels = model.config.id2label

        filtered = [
            {"label": labels[i], "confidence": p.item()}
            for i, p in enumerate(probs)
            if p.item() > 0.2
        ]

        total = sum(item["confidence"] for item in filtered)
        if total == 0:
            return {"html": "<p>No waste types detected with sufficient confidence.</p>"}

        # Calculate percentages
        composition = [
            {
                "label": item["label"],
                "percentage": round((item["confidence"] / total) * 100, 2)
            }
            for item in filtered
        ]

        # Generate HTML pie chart using Chart.js
        labels_js = [item["label"] for item in composition]
        data_js = [item["percentage"] for item in composition]

        html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h3>Waste Composition Analysis</h3>
            <canvas id="wasteChart" width="400" height="400"></canvas>
            <script>
                const ctx = document.getElementById('wasteChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'pie',
                    data: {{
                        labels: {labels_js},
                        datasets: [{{
                            data: {data_js},
                            backgroundColor: [
                                '#36A2EB',
                                '#FF6384',
                                '#FFCE56',
                                '#8BC34A',
                                '#9C27B0',
                                '#FF9800'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return {"html": html}

    except Exception as e:
        return {"error": str(e)}
