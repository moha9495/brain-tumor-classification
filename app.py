import io
import base64

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image

# ---------------------------------------------------------------
# 1. Settings — must match what we used in Colab
# ---------------------------------------------------------------
MODEL_PATH = "brain_tumor_model.pt"
CLASSES = ["glioma", "meningioma", "pituitary"]   # same order as ImageFolder in Colab

# Same "eval" recipe as Lesson 2 (no augmentation when predicting)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------
# 2. Rebuild the exact same model shape, then load the trained weights
# ---------------------------------------------------------------
def load_model():
    model = models.resnet50(weights=None)          # empty ResNet50 (no download needed)
    model.fc = nn.Sequential(                      # same head as Lesson 3
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3),
    )
    state = torch.load(MODEL_PATH, map_location="cpu")   # trained on GPU, running on CPU
    model.load_state_dict(state)
    model.eval()                                   # prediction mode: dropout off
    return model


print("Loading model...")
model = load_model()
print("Model ready.")


# ---------------------------------------------------------------
# 3. Predict one image
# ---------------------------------------------------------------
def predict(image):
    x = transform(image).unsqueeze(0)              # [3,224,224] -> [1,3,224,224] (a batch of 1)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]  # scores -> probabilities that add up to 1
    scores = {c: float(p) for c, p in zip(CLASSES, probs)}
    best = max(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------
# 4. The web app
# ---------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None

    if request.method == "POST":                   # the user pressed "Analyse"
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please choose an image first."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
            except Exception:
                error = "That file isn't an image I can read. Try a .jpg or .png."
            else:
                label, scores = predict(image)

                # turn the uploaded image into text so the page can show it back
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                preview = base64.b64encode(buf.getvalue()).decode()

                result = {
                    "label": label,
                    "confidence": scores[label],
                    "scores": sorted(scores.items(), key=lambda kv: -kv[1]),
                    "preview": preview,
                }

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
