# Troubleshooting Guide

Common issues, error messages, and solutions for Brain Tumour Detection.

> The existing `docs/troubleshooting.md` (now replaced by this file) covered container issues. This version is expanded to cover all service layers.

---

## Table of Contents

1. [Container Issues](#container-issues)
2. [AI Service Issues](#ai-service-issues)
3. [Backend Issues](#backend-issues)
4. [Frontend Issues](#frontend-issues)
5. [Authentication Issues](#authentication-issues)
6. [Training Issues](#training-issues)
7. [Inference Issues](#inference-issues)
8. [Dataset Issues](#dataset-issues)
9. [Performance Issues](#performance-issues)
10. [Build Issues](#build-issues)
11. [Network Issues](#network-issues)
12. [Diagnostic Commands](#diagnostic-commands)

---

## Container Issues

### AI service container keeps restarting

**Symptoms:** `docker compose ps` shows `ai-service` as `Restarting`

**Cause 1 — JWT_SECRET_KEY is still the default:**
```
[AI-SERVICE] ERROR: JWT_SECRET_KEY is still the default placeholder!
```
Fix: Set a strong JWT secret in `ai-service/.env`:
```bash
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> ai-service/.env
docker compose restart ai-service
```

**Cause 2 — Port 8000 already in use:**
```bash
# Linux / macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```
Fix: Stop the conflicting process or change `AI_SERVICE_PORT` in `.env`.

---

### Container is unhealthy

**Check health details:**
```bash
docker inspect bt_ai_service | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(json.dumps(d[0]['State']['Health'], indent=2))"
```

The AI service `start_period` is **30 seconds** — allow this time for the PyTorch/MambaVision model registry to initialise before declaring it broken.

---

### "No space left on device"

```bash
docker system df           # show space usage
docker system prune -f     # remove stopped containers + dangling images
# WARNING: never prune named volumes on production
docker volume prune -f --filter "label!=keep"
```

---

### Services can't reach each other inside Docker

Containers must reference each other by service name, not `localhost`:

| From | To | Correct URL |
|---|---|---|
| backend | ai-service | `http://ai-service:8000` |
| frontend | backend | `http://backend:5000` |
| nginx | backend | `http://backend:5000` |

Check `AI_SERVICE_URL` in `backend/.env` — should be `http://ai-service:8000`, not `http://localhost:8000`.

---

## AI Service Issues

### "No model available" when predicting

**Symptom:** 404 response with `MODEL_NOT_FOUND`

**Cause:** No trained weights exist for the requested model.

**Fix:** Train the model first:
```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mambavision", "epochs": 30}'
```

Or check what models are available:
```bash
curl http://localhost:8000/api/v1/health | python3 -m json.tool | grep -A5 "models_available"
```

---

### AI service starts but returns 500 on all requests

Check the logs:
```bash
# Docker
docker compose logs ai-service

# Local
cat ai-service/logs/$(date +%Y-%m-%d).log
```

**Common causes:**
- Missing `.env` file — copy from `.env.example`
- Invalid `ACTIVE_MODEL` value — must be one of `mambavision`, `cnn`, `vgg16`, `resnet50`, `efficientnet`
- `DATASET_RAW_DIR` points to a non-existent path (directories are auto-created but check permissions)

---

### Grad-CAM returns blank/black image

**Causes and fixes:**
1. **Model not fully trained** — a model with random weights produces meaningless Grad-CAM. Train for more epochs.
2. **Wrong layer name** — the Grad-CAM implementation targets the last convolutional layer. If you added a custom architecture, verify the layer name in `app/utils/gradcam.py`.
3. **Very low confidence prediction** — check the probability scores. If all probabilities are near 0.25, the model is uncertain.

---

## Backend Issues

### Backend fails to start — "SQLITE_CANTOPEN"

```
Error: SQLITE_CANTOPEN: unable to open database file
```

**Fix:** Ensure the database directory exists and run migrations:
```bash
mkdir -p backend/database
cd backend && node database/migrate.js
```

---

### Upload fails with 413 (Request Entity Too Large)

**nginx fix** — increase `client_max_body_size` in nginx config:
```nginx
client_max_body_size 50M;
```

**Express fix** — already handled via multer; check `UPLOAD_DIR` exists and has write permissions.

---

### Backend can't reach the AI service

```
Error: connect ECONNREFUSED 127.0.0.1:8000
```

**Fixes:**
- In Docker: set `AI_SERVICE_URL=http://ai-service:8000` (not localhost)
- Locally: confirm the AI service is running on port 8000
- Check for firewall rules blocking inter-process communication

---

## Frontend Issues

### Blank page after `npm run dev`

1. Check the browser console for errors
2. Verify `VITE_API_BASE_URL` in `frontend/.env.local` points to the running backend
3. Check for TypeScript compilation errors: `cd frontend && npm run type-check`

---

### "Network Error" when submitting a prediction

**Cause:** CORS is blocking the request.

**Fixes:**
- Backend must list the frontend origin in `FRONTEND_URL`
- AI service must list the frontend origin in `ALLOWED_ORIGINS`
- In development: ensure both services are running before opening the frontend

Check CORS headers:
```bash
curl -v -X OPTIONS http://localhost:8000/api/v1/predict \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST"
```
You should see `Access-Control-Allow-Origin: http://localhost:3000` in the response headers.

---

### Grad-CAM image doesn't display

1. Open browser DevTools → Network tab
2. Check the `/predict` response — does `gradcam_b64` contain a data URL?
3. If missing: confirm `VITE_ENABLE_GRADCAM=true` in `frontend/.env.local`

---

### Frontend build fails in CI

```
error TS2307: Cannot find module '...'
```

Fix: Run `npm ci` (not `npm install`) to use the exact locked versions:
```bash
cd frontend && npm ci
```

---

## Authentication Issues

### 401 Unauthorised on every request

**Token expired:** Access tokens expire after 30 minutes. Refresh:
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<your-refresh-token>"}'
```

**Wrong header format:** Must be `Authorization: Bearer <token>` (with capital B and a space after Bearer).

---

### 403 Forbidden

Your user account has insufficient role for the endpoint.

Check your role:
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/auth/me
```

Contact an Admin to upgrade your role.

---

### Account locked after failed logins

After 5 consecutive failed logins the account is locked for 15 minutes.

**Wait 15 minutes**, or ask an Admin to unlock immediately:
```bash
curl -X POST http://localhost:8000/api/v1/auth/users/{user_id}/unlock \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

### JWT_SECRET_KEY error on startup

```
pydantic_core.InitErrorDetails: Value error, jwt_secret_key is the default
```

The AI service refuses to start if `JWT_SECRET_KEY` is the default value `change-me-in-production-...`.

Fix:
```bash
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> ai-service/.env
```

---

## Training Issues

### Training fails with "Dataset not prepared"

**Fix:** Prepare the dataset before training:
```bash
curl -X POST http://localhost:8000/api/v1/dataset/prepare \
  -H "Content-Type: application/json" \
  -d '{"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}'
```

---

### Training OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

**Fixes:**
1. Reduce `batch_size` (try 16 or 8)
2. Use a smaller model (`cnn` instead of `mambavision`)
3. Reduce `image_size` to 160 or 128 (changes model input shape)
4. Enable PyTorch memory-efficient attention: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `ai-service/.env`

---

### Async training job stuck in "pending"

**Cause:** The background task thread died silently.

**Check logs:**
```bash
grep -i "training\|job\|error" ai-service/logs/$(date +%Y-%m-%d).log | tail -20
```

**Fix:** Restart the AI service — pending jobs are cleared on restart. Then re-submit via `POST /train/start`.

---

### Training accuracy stuck at 25%

The model is predicting the same class for all inputs (class imbalance or bad data).

**Checks:**
1. Verify dataset class balance: `GET /api/v1/dataset/info`
2. Check class weights are being applied (logged at training start)
3. Try a lower learning rate: `"learning_rate": 0.00001`
4. Verify images are not all identical or corrupted: `POST /preprocess/quality-check`

---

## Inference Issues

### Very slow inference (> 500ms)

**Expected latency:** ~45-80ms with a loaded model on CPU. > 500ms indicates a cache miss (model load).

The first prediction after startup always loads the model (~2-5 seconds). Subsequent predictions use the LRU cache and should be fast.

**Check cache stats:**
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/cache
```

If hit rate is below 90%, increase the cache size or reduce `ACTIVE_MODEL` changes.

---

### Predictions are wrong / low confidence

1. **Check image orientation** — the model was trained on standard axial/sagittal/coronal MRI views
2. **Run quality check first** — `POST /preprocess/quality-check` will flag blur or low contrast
3. **Model needs more training** — evaluate accuracy with `POST /evaluate`
4. **Wrong model loaded** — check `active_model` in `GET /health`

---

## Dataset Issues

### Dataset validation fails

```json
{"valid": false, "errors": ["Missing class directory: notumor in Training/"]}
```

Ensure your dataset matches the expected structure:
```
dataset/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    └── (same four directories)
```

Class directory names must match the `CLASS_NAMES` config exactly (default: `glioma,meningioma,notumor,pituitary`).

---

### "dataset_info.json not found"

Run dataset prepare first:
```bash
curl -X POST http://localhost:8000/api/v1/dataset/prepare \
  -H "Content-Type: application/json" \
  -d '{"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}'
```

---

## Performance Issues

### High CPU usage during idle

**Cause:** Metrics collection polling or benchmark running in background.

Check for running benchmarks:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/benchmark/result
```

If `running: true`, wait for it to complete.

---

### Memory keeps growing

Check for memory leaks:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/performance/memory
```

Look for `warning_count > 0` and `operations` with large `delta_mb`.

**Common fix:** The model cache holds loaded weights in RAM. This is expected and intentional. If RAM is too constrained, reduce the LRU cache size by setting a lower limit in `app/inference/cache.py`.

---

## Build Issues

### `npm ci` fails with peer dependency errors

```bash
# Use legacy peer deps flag
npm ci --legacy-peer-deps
```

Or delete `node_modules` and `package-lock.json` and run `npm install` to regenerate the lockfile.

---

### Docker build fails — "failed to solve: Dockerfile parse error"

Ensure Docker BuildKit is enabled (required for multi-stage builds):
```bash
export DOCKER_BUILDKIT=1
docker compose build
```

---

## Network Issues

### Services on different ports can't communicate locally

If running services locally (not in Docker), each service must use `localhost` with the correct port:
- Frontend → Backend: `VITE_API_BASE_URL=http://localhost:5000`
- Frontend → AI Service: `VITE_AI_SERVICE_URL=http://localhost:8000`
- Backend → AI Service: `AI_SERVICE_URL=http://localhost:8000`

---

### Rate limit hit (429 Too Many Requests)

The `Retry-After` response header tells you how many seconds to wait.

| Endpoint | Limit | Reset |
|---|---|---|
| `POST /auth/login` | 5 / min | 60s |
| `POST /predict` | 60 / min | 60s |
| `POST /train` | 5 / min | 60s |

For development / testing, increase limits in `ai-service/app/security/rate_limit.py`.

---

## Diagnostic Commands

### Complete system health check

```bash
# Check all services
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
curl -s http://localhost:5000/api/health   | python3 -m json.tool
curl -s http://localhost:3000              | head -5

# Check container status
docker compose ps

# Check container resource usage
docker stats --no-stream

# Check disk space
df -h
docker system df
```

### Collect diagnostic logs

```bash
# Capture last 500 lines from all services
docker compose logs --tail=500 > diagnostic-$(date +%Y%m%d-%H%M%S).log
```

### Test authentication end-to-end

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: ${TOKEN:0:20}..."
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/auth/me
```

### Validate environment files

```bash
bash scripts/validate-env.sh
```

### Run tests

```bash
# AI service tests only
cd ai-service && python -m pytest tests/ -v --tb=short 2>&1 | tail -20

# All tests
make test
```
