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

        topk = torch.topk(probs, k=3)
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

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label

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

        # Build the chart HTML
        html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h3>Waste Composition (Top Predictions)</h3>
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
                                'rgba(153, 102, 255, 0.6)'
                            ],
                            borderColor: [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 206, 86, 1)',
                                'rgba(75, 192, 192, 1)',
                                'rgba(153, 102, 255, 1)'
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
