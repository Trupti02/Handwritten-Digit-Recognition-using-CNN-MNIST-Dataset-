import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import pickle
import traceback
import base64
from io import BytesIO
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("model/label_map.pkl", "rb") as f:
    id_to_label = pickle.load(f)
print(f"Loaded {len(id_to_label)} classes: {id_to_label}")


class HandwrittenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


print("Loading CNN model...")
model = HandwrittenCNN().to(DEVICE)
model.load_state_dict(torch.load("model/cnn_model.pth", map_location=DEVICE))
model.eval()
print("Model ready!")


def preprocess_image(img):
    img    = img.convert('L')           # Grayscale
    img    = ImageOps.invert(img)       # Invert: white digit on black
    img    = img.resize((28, 28), Image.LANCZOS)
    arr    = np.array(img, dtype=np.float32) / 255.0
    arr    = (arr - 0.1307) / 0.3081   # MNIST normalization
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = None

    
    data = request.get_json(silent=True)
    if data and 'image' in data:
        try:
            img_bytes = base64.b64decode(data['image'].split(',')[1])
            img = Image.open(BytesIO(img_bytes))
            print("Input source: Canvas drawing")
        except Exception as e:
            return jsonify({'error': f'Canvas decode error: {e}'}), 400

    
    elif 'file' in request.files:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'Empty file uploaded'}), 400
        fname    = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(filepath)
        img = Image.open(filepath)
        print(f"Input source: File upload -> {filepath}")

    else:
        return jsonify({'error': 'No image provided. Send JSON with image key or multipart file.'}), 400

    
    try:
        tensor = preprocess_image(img)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0].cpu().numpy()

        top5_idx = probs.argsort()[-5:][::-1]

        result = {
            'prediction': id_to_label.get(int(top5_idx[0]), '?'),
            'confidence': round(float(probs[top5_idx[0]]) * 100, 2),
            'top5': [
                {
                    'label':      id_to_label.get(int(i), str(i)),
                    'confidence': round(float(probs[i]) * 100, 2)
                }
                for i in top5_idx
            ],
            'all_probs': [round(float(p) * 100, 2) for p in probs]
        }
        print(f"Prediction: {result['prediction']}  Confidence: {result['confidence']}%")
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)