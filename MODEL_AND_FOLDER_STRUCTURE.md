# Models Used

## AI Service (Python / TensorFlow-Keras)

### Shipped model (saved in repo)
- **Model key/name**: `efficientnet`
- **Backbone**: EfficientNetB3 (`weights="imagenet"`)
- **Saved artifact**: [ai-service/saved_models/efficientnet/efficientnet.keras](file:///d:/PROJECT/BrainTumor/ai-service/saved_models/efficientnet/efficientnet.keras)
- **Checkpoints**:
  - [best_phase1.weights.h5](file:///d:/PROJECT/BrainTumor/ai-service/saved_models/efficientnet/checkpoints/best_phase1.weights.h5)
  - [best_phase2.weights.h5](file:///d:/PROJECT/BrainTumor/ai-service/saved_models/efficientnet/checkpoints/best_phase2.weights.h5)
- **Metadata**: [model_info.json](file:///d:/PROJECT/BrainTumor/ai-service/saved_models/efficientnet/model_info.json)
- **Architecture factory**: [architectures.py](file:///d:/PROJECT/BrainTumor/ai-service/app/models/architectures.py)
- **Inference loader**: [load_model.py](file:///d:/PROJECT/BrainTumor/ai-service/app/models/load_model.py)

### Supported architectures (no saved weights committed)
- `cnn` (custom)
- `vgg16` (ImageNet weights at runtime)
- `resnet50` (ImageNet weights at runtime)
- `efficientnet` (ImageNet weights at runtime)

## Backend (Node)
- Classifier code includes stubs (not production-ready model artifacts committed):
  - [ednSvm.js](file:///d:/PROJECT/BrainTumor/backend/pipeline/classifier/ednSvm.js)
  - [modelSerializer.js](file:///d:/PROJECT/BrainTumor/backend/pipeline/classifier/modelSerializer.js)

# Folder Structure (Top Level)

```text
BrainTumor/
  .github/                GitHub Actions workflows
  ai-service/             Python AI service (training + inference)
    app/                  API, inference pipeline, models, preprocessing
    saved_models/         Saved Keras models (includes efficientnet/)
    scripts/              Utilities (prepare/verify)
    tests/                Pytest suite
  backend/                Node/Express API + classic image pipeline
  braintumor-main/        Full stack copy (ai-service + backend + frontend + docs + docker + dataset)
  README.md
  LICENSE
  CHANGELOG.md
  Makefile
  SECURITY.md
  VERSION
```

