# AgroVision
AI-powered crop leaf disease screening using deep learning, transfer learning, and confidence-based expert referral.

# AgroVision 🌿

AgroVision is an AI-powered crop leaf screening system that uses deep learning to identify supported plant health conditions from leaf images. Users can upload a photograph of a crop leaf, and the trained model predicts the corresponding healthy or disease class along with its confidence score.

The system uses transfer learning with models such as MobileNetV2 or EfficientNet to enable efficient image classification. It incorporates realistic data augmentation and field-oriented evaluation to improve model robustness under practical conditions. Predictions with low confidence or unsupported cases are flagged for expert review rather than providing potentially unreliable results.

The application is designed with **FastAPI** for model serving and **Docker** for reproducible deployment, providing a lightweight and scalable foundation for AI-assisted crop health screening.

### Key Features
- 🌱 Crop leaf image upload and classification
- 🧠 Deep-learning-based disease screening
- 📊 Predicted class and confidence score
- 🔎 Confidence-based referral for uncertain predictions
- 🧪 Data augmentation and realistic model evaluation
- ⚡ FastAPI-based inference API
- 🐳 Docker-based deployment

### Objective
To develop a practical, reproducible AI screening tool that supports preliminary crop health assessment while recognizing the limitations of automated predictions and the importance of expert verification.

**Note:** AgroVision is intended as a screening aid, not a replacement for professional agricultural diagnosis. Its predictions are limited to the classes supported by the trained model.
