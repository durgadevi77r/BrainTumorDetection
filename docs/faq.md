# Frequently Asked Questions

Brain Tumour Detection — answers to the most common questions.

---

## General

**Q: What does this application actually do?**

It classifies MRI brain scans into one of four categories: glioma, meningioma, pituitary tumour, or no tumour. It uses deep learning (MambaVision-T by default) trained on labelled MRI image datasets. Grad-CAM heatmaps show which image regions influenced the classification.

---

**Q: Is this a medical diagnostic tool?**

No. This application is an educational and research tool for exploring deep learning applied to medical imaging. It is **not validated for clinical use** and must not be used to make real medical decisions. All clinical diagnoses must be performed by qualified medical professionals.

---

**Q: What image formats are supported?**

JPEG, PNG, BMP, and TIFF. Images are automatically resized to 224×224 pixels during preprocessing. Very small images (under 32×32) or extremely large ones (over 4096×4096) may produce poor quality check scores.

---

**Q: How accurate is the model?**

MambaVision-T trained on the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) typically achieves **97–99% validation accuracy**. Accuracy on out-of-distribution data (different scanner protocols, patient populations, or image quality) will be lower and should be evaluated separately.

---

**Q: Can I use my own dataset?**

Yes. Organise your images in the required structure (one subdirectory per class inside `Training/` and `Testing/`), point `DATASET_RAW_DIR` to your dataset, update `CLASS_NAMES` in `.env` if your classes differ, and run `POST /dataset/validate` then `POST /dataset/prepare` before training.

---

## Installation & Setup

**Q: Do I need a GPU?**

No. The application runs on CPU. A GPU dramatically speeds up training (10–100×) but inference is fast enough on a modern CPU (~50ms per image). GPU support requires NVIDIA hardware with CUDA 12.x — see [Installation Guide](installation.md#gpu-support-optional).

**Q: Which Python version is required?**

Python 3.12 is recommended. PyTorch 2.4+ supports Python 3.8–3.12. Python 3.13 support may require testing specific PyTorch nightly builds.

**Q: Can I run this on Windows?**

Yes. Docker Desktop works on Windows 10/11. Local development (without Docker) works with Python 3.12 and Node.js 20 installed. PowerShell setup scripts (`setup_env.ps1`, `deploy.ps1`, `validate-env.ps1`) are provided for Windows.

**Q: Can I run this on Apple Silicon (M1/M2/M3)?**

Yes. PyTorch has native Apple Silicon support:
1. PyTorch will automatically use MPS (Metal Performance Shaders) backend when available
2. Docker images are built for `linux/amd64` — use Rosetta 2 or rebuild natively with `--platform linux/arm64`

---

## Training

**Q: How long does training take?**

On CPU, 30 epochs of MambaVision-T on ~5,700 images takes approximately 60–120 minutes. On a mid-range GPU (RTX 3060), the same run takes about 8–15 minutes. VGG16 takes 2–4× longer than MambaVision due to its larger parameter count.

**Q: Which model architecture should I use?**

Start with **MambaVision-T** (the default). It offers state-of-the-art accuracy using vision Mamba architecture with selective state space models. Use **EfficientNet-B3** for a lighter alternative with excellent accuracy-to-speed tradeoff. Use **Custom CNN** for rapid experimentation or very limited hardware. Avoid **VGG16** unless you have a specific reason — it uses 138M parameters and trains slowly.

**Q: What do the training hyperparameters do?**

| Parameter | Effect |
|---|---|
| `epochs` | More epochs → potentially better accuracy but longer training + risk of overfitting |
| `batch_size` | Larger batches → faster training but more RAM; smaller → better generalisation |
| `learning_rate` | Lower → more stable but slower convergence; default 0.0001 works well |
| `fine_tune` | Two-phase training: head first, then backbone — almost always improves final accuracy |
| `fine_tune_layers` | More layers unfrozen → more capacity but risk of catastrophic forgetting |

**Q: Can I resume a failed training job?**

Not automatically. PyTorch checkpoints save the best weights so far (`best.pt` or HuggingFace format). After a failure, restart training — it will overwrite the checkpoint only if the new run achieves a better validation accuracy.

**Q: How do I prevent overfitting?**

1. Enable data augmentation (handled automatically in `app/preprocessing/augmentation.py`)
2. Use dropout (built into all four architectures)
3. Apply early stopping (built into the training loop — patience=10 on val_loss)
4. Reduce model complexity (use CNN instead of VGG16 for small datasets)
5. Ensure your dataset is large enough (>500 images per class recommended)

---

## Inference

**Q: Why is the first prediction slow?**

The first prediction after a service restart loads the model weights from disk into memory (~3–7 seconds for MambaVision). Subsequent predictions use the LRU model cache and are fast (~45–80ms).

**Q: Can the model handle non-MRI images?**

Technically yes — it will produce a prediction for any image. However, the result will be meaningless. The model was trained exclusively on MRI scans and has no concept of what "not an MRI" looks like. Use the quality check endpoint to filter obviously bad inputs.

**Q: How confident does the prediction need to be to trust it?**

As a rough guide:
- **> 90% confidence** — strong prediction, Grad-CAM should show a clear region of interest
- **70–90%** — reasonable but examine the Grad-CAM carefully
- **< 70%** — uncertain; the image quality may be poor or the scan may be unusual

These are not clinical thresholds — they are informal engineering heuristics.

**Q: What is batch inference?**

Batch inference processes multiple images in parallel using a thread pool. Submit a ZIP archive or multiple files to `POST /predict/batch`. Results include per-image predictions and confidence scores.

---

## Grad-CAM

**Q: What is Grad-CAM?**

Gradient-weighted Class Activation Mapping (Grad-CAM) is an explainability technique that highlights which parts of the image the neural network focused on when making its prediction. Warm (red/yellow) areas contributed most to the classification.

**Q: Why does Grad-CAM sometimes highlight the wrong area?**

1. The model may not have learned the right features if training data is limited
2. Image contrast or quality issues may mislead the network
3. The particular scan may be genuinely ambiguous
4. Pre-processing artefacts (from skull stripping or normalisation) can affect attention patterns

Grad-CAM is an explanation tool, not a guarantee of correct reasoning.

**Q: Can I save the Grad-CAM image?**

Yes. The `gradcam_b64` field in the prediction response is a base64-encoded PNG. Decode it:

```python
import base64, json

response = json.loads(...)   # your prediction response
img_data = response['data']['gradcam_b64'].split(',')[1]
with open('gradcam.png', 'wb') as f:
    f.write(base64.b64decode(img_data))
```

Grad-CAM images are also saved to `ai-service/gradcam_output/`.

---

## Security & Authentication

**Q: Do I need to log in to use the application?**

In development mode (`PREDICTION_AUTH_MODE=public`), the predict and health endpoints are publicly accessible. All other endpoints require authentication. In production, set `PREDICTION_AUTH_MODE=authenticated` to require a token for inference too.

**Q: How do I create the first admin user?**

The first admin account is seeded from the `app/security/auth.py` user store initialisation. Check the `UserStore._seed_default_users()` method for the default credentials and change them immediately.

**Q: How long do tokens last?**

- Access token: **30 minutes** (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Refresh token: **7 days** (configurable via `REFRESH_TOKEN_EXPIRE_DAYS`)

**Q: What happens to tokens after logout?**

The logout endpoint adds the token's JTI (JWT ID) to an in-process revocation set. Revoked tokens are rejected on subsequent requests. In a multi-instance deployment, replace this with a shared Redis store.

**Q: How do I change the JWT secret?**

1. Update `JWT_SECRET_KEY` in `ai-service/.env`
2. Restart the AI service
3. All existing tokens are immediately invalidated — all users must log in again

---

## API & Integration

**Q: Where is the interactive API documentation?**

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

**Q: Can I call the AI service directly from the frontend?**

Yes. Set `VITE_AI_SERVICE_URL` in `frontend/.env.local`. The AI service has CORS configured to allow the frontend origin.

**Q: How do I integrate with an external system?**

Use the REST API. Authenticate with `POST /api/v1/auth/login` to obtain a token, then include `Authorization: Bearer <token>` on all requests. See [API Reference](api_reference.md) for full endpoint documentation.

**Q: Is there an OpenAPI spec I can import into Postman?**

Yes. Import: http://localhost:8000/openapi.json

---

## Docker & Deployment

**Q: Can I run without Docker?**

Yes — see [Option B in the Installation Guide](installation.md#option-b--local-development).

**Q: How do I update to a new version?**

```bash
git pull origin main
docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up --build -d
```

Or use the deploy script: `bash scripts/deploy.sh production`

**Q: How do I back up my trained models?**

```bash
bash scripts/backup.sh
```

This creates a timestamped archive in `backups/` that includes `models_volume`, `dataset_volume`, `db_volume`, and `uploads_volume`.

**Q: The Docker build takes too long. How do I speed it up?**

1. Docker layer caching — don't change `requirements.txt` or `package.json` unless necessary
2. Use `--no-cache` only when you suspect a stale layer
3. Use a Docker registry (GHCR) so CI workers pull pre-built layers

---

## Testing

**Q: How do I run all the tests?**

```bash
make test
# or from each service directory:
cd ai-service && python -m pytest tests/ -v
cd backend    && npm test
cd frontend   && npm test
```

**Q: How many tests are there?**

1,492+ tests across all three services:
- AI Service (pytest): 1,100+
- Frontend (Vitest): 280+
- Backend (Jest): 112+

**Q: Why do some AI service tests require a trained model?**

Tests marked with `@pytest.mark.slow` or that call the `/predict` endpoint directly need model weights. These are skipped in CI unless model weights are present. Use `pytest -m "not slow"` to skip them locally.
