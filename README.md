# 🌿 Plant Deficiency Detection App

A mobile-first system that allows users to take a picture of a plant, detect nutrient deficiencies using a deep learning model (DenseNet), and receive actionable recommendations for plant health improvement.

---

## 📱 Features

- Capture plant leaf images directly from the mobile app.
- Send images to a FastAPI-powered backend.
- Run real-time inference using a TensorFlow DenseNet model.
- Return a list of detected deficiencies (e.g., Nitrogen, Iron).
- Suggest remedies or fertilizers to address the deficiencies.

---

## 🧠 Machine Learning Model

- **Architecture**: DenseNet (Dense Convolutional Network)
- **Framework**: TensorFlow 2.x
- **Input**: Plant leaf images (e.g., 224x224 RGB)
- **Output**: One or more predicted deficiency classes
- **Training Dataset**: A labeled dataset of plant leaves with annotated deficiencies (e.g., PlantVillage or a custom dataset)

---

## 📦 Tech Stack

| Component    | Technology        |
|--------------|-------------------|
| Frontend     | Swift (iOS app with camera) |
| Backend      | FastAPI (Python)  |
| ML Pipeline  | TensorFlow + Keras |
| Hosting      | ngrok (for local testing), or deploy to Render / Railway / AWS |
| Format       | `multipart/form-data` for image upload |

---

## 🧪 Inference Pipeline (Backend)

1. **Receive Image** (via POST `/predict/`)
2. **Preprocess Image**
   - Resize to 224x224
   - Normalize pixel values
3. **Model Inference**
   - Load trained DenseNet model
   - Predict deficiency class(es)
4. **Return Response**
   - JSON containing `deficiencies` and `recommendations`

Example response:
```json
{
  "plant": "Tomato",
  "deficiencies": ["Nitrogen", "Iron"],
  "advice": "Apply a balanced NPK fertilizer with chelated iron weekly."
}
