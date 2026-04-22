import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import pickle
import traceback
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, render_template, request, jsonify
from scipy import ndimage

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("model/label_map.pkl", "rb") as f:
    id_to_label = pickle.load(f)

class HandwrittenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)

print("Loading model...")
model = HandwrittenCNN().to(DEVICE)
model.load_state_dict(torch.load("model/cnn_model.pth", map_location=DEVICE))
model.eval()
print("Model ready!")

# ── FIXED PREPROCESSING ──────────────────────────────────────────────────────
def preprocess_image(img):
    # Step 1: Convert to grayscale
    img = img.convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Step 2: Check if background is light or dark
    # Canvas = black background, white digit → do NOT invert
    # Uploaded image = white background, dark digit → DO invert
    mean_val = img_array.mean()
    if mean_val > 127:
        # Light background (uploaded image) — invert to get white digit on black
        img_array = 255.0 - img_array

    # Step 3: Remove near-zero noise
    img_array[img_array < 30] = 0

    # Step 4: Crop tightly around the digit using bounding box
    rows = np.any(img_array > 30, axis=1)
    cols = np.any(img_array > 30, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]

    # Step 5: Add padding (20% on each side) so digit doesn't touch edges
    h, w = img_array.shape
    pad = int(max(h, w) * 0.20)
    img_array = np.pad(img_array, pad, mode='constant', constant_values=0)

    # Step 6: Resize to 28x28 using PIL (high quality)
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(pil_img, dtype=np.float32)

    # Step 7: Normalize to 0-1
    img_array = img_array / 255.0

    # Step 8: MNIST normalization (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081

    # Step 9: Convert to tensor → shape (1, 1, 28, 28)
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = None

    # Handle canvas drawing (base64)
    data = request.get_json(silent=True)
    if data and 'image' in data:
        try:
            img_bytes = base64.b64decode(data['image'].split(',')[1])
            img = Image.open(BytesIO(img_bytes))
            print("Source: Canvas drawing")
        except Exception as e:
            return jsonify({'error': f'Canvas error: {e}'}), 400

    # Handle file upload
    elif 'file' in request.files:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'Empty file'}), 400
        fname    = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(filepath)
        img = Image.open(filepath)
        print(f"Source: File upload")
    else:
        return jsonify({'error': 'No image provided'}), 400

    try:
        tensor = preprocess_image(img)

        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        top5_idx = probs.argsort()[-5:][::-1]
        result   = {
            'prediction': id_to_label.get(int(top5_idx[0]), '?'),
            'confidence': round(float(probs[top5_idx[0]]) * 100, 2),
            'top5': [
                {'label':      id_to_label.get(int(i), str(i)),
                 'confidence': round(float(probs[i]) * 100, 2)}
                for i in top5_idx
            ],
            'all_probs': [round(float(p) * 100, 2) for p in probs]
        }
        print(f"Prediction: {result['prediction']} ({result['confidence']}%)")
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)