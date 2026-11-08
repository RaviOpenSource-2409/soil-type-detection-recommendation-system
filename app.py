from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# ===== Load Model =====
model = None

def load_model():
    global model
    try:
        from tensorflow import keras
        model = keras.models.load_model("soil_final_model.h5")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model load error: {e}")

# ===== Class Names (same as training order) =====
class_names = [
    "Alluvial", "Arid", "Black", "Clay", "Dry", "Laterite",
    "Loamy", "Peat", "Red", "Sandy", "Sandy_loam", "Yellow"
]

CONFIDENCE_THRESHOLD = 0.55  # 55% threshold

# ===== Predict Route =====
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        import io
        from PIL import Image
        from tensorflow.keras.preprocessing import image as keras_image

        # Read image from memory
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        confidence = float(np.max(prediction))
        soil_class = class_names[int(np.argmax(prediction))]

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'is_soil': False,
                'soil_type': None,
                'confidence': round(confidence, 4),
                'reason': 'मॉडल को यह मिट्टी की फोटो नहीं लग रही या छवि अस्पष्ट है'
            })
        else:
            return jsonify({
                'is_soil': True,
                'soil_type': soil_class,
                'confidence': round(confidence, 4),
                'reason': f'{soil_class} मिट्टी पहचानी गई'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== Serve HTML Frontend =====
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    load_model()
    print("\n🌾 Soil Detection Server चालू हो रहा है...")
    print("🌐 Browser में खोलें: http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
