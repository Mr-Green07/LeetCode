"""Run a real chest X-ray through the pipeline and inspect what the model saw."""
import sys

import httpx

sys.path.insert(0, ".")
from app.ml.xray.preprocess import build_model_input, preprocess_xray_array  # noqa: E402
from app.ml.xray.features import extract_features, feature_vector  # noqa: E402

PATH = (
    r"C:\Users\yuvra\.minimax\v2\assets\2026\10\29\16-16-51-721-asset_20261029"
    r"-161651-721_0c76183f626a_9d3794a9-2ed8fb4f-d1b5-4e0f-bd35-e553cb908510.jpg"
)

client = httpx.Client(base_url="http://127.0.0.1:8000/api/v1", timeout=180)

with open(PATH, "rb") as handle:
    response = client.post(
        "/xray/analyze",
        files={"file": ("chest.jpg", handle, "image/jpeg")},
        data={"patient_id": "REAL-1", "age": 45, "sex": "male"},
    )
print("HTTP", response.status_code)
payload = response.json()

prediction = payload.get("prediction", {})
print("prediction:", prediction.get("primary_class"), round(prediction.get("confidence", 0), 4))
print("all classes:", {k: round(v, 4) for k, v in (prediction.get("probabilities") or {}).items()})
print("findings:", [f.get("name") for f in payload.get("findings", [])])
print("model_source:", payload.get("model_source"))

# What does the synthetic-trained network actually see?
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

image = Image.open(PATH).convert("L")
array = np.asarray(image, dtype=np.float32) / 255.0
pre = preprocess_xray_array(array)
print("\npreprocess: shape", pre.image.shape, "dtype", pre.image.dtype,
      "min", float(pre.image.min()), "max", float(pre.image.max()))
print("mean intensity:", float(pre.image.mean()))

features = extract_features(pre)
vector = feature_vector(features)
names = list(features.keys())
print("\nfeature vector (name, value):")
for name, value in zip(names, vector):
    print(f"  {name:34s} {value: .4f}")
