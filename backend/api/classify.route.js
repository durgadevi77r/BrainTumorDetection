'use strict';

/**
 * POST /api/classify/:imageId
 *
 * Forwards the raw MRI image to the FastAPI AI service (POST /api/v1/predict),
 * stores the real prediction in the `results` table, and returns the full
 * AI response including class, confidence, per-class probabilities, and the
 * Grad-CAM heatmap path.
 *
 * Request: no body — imageId from URL identifies which image to classify.
 *
 * Success response 200:
 *  {
 *    success: true,
 *    data: {
 *      image_id, predicted_class, confidence, probabilities,
 *      gradcam_url, model_used, computational_time_ms
 *    }
 *  }
 */

const express  = require('express');
const { v4: uuidv4 } = require('uuid');
const fs       = require('fs');
const path     = require('path');
const FormData = require('form-data');
const axios    = require('axios');
const router   = express.Router();

const db       = require('../database/db');
const config   = require('../config');
const { validateImageId } = require('../middleware/validateRequest');
const { startTimer }      = require('../utils/timer');
const logger   = require('../utils/logger');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

/**
 * Query the AI service health endpoint and return the name of the best
 * available (trained) model to use for classification.
 *
 * Priority: whatever settings.active_model is set to on the AI side — but
 * only if weights exist for it.  If not, fall back to the first architecture
 * that does have weights.  Ultimately falls back to 'cnn'.
 *
 * The result is cached for 60 s so we don't hit /health on every request.
 */
let _modelCache   = null;
let _modelCacheTs = 0;
const MODEL_CACHE_TTL_MS = 60_000;

async function getBestAvailableModel() {
  const now = Date.now();
  if (_modelCache && (now - _modelCacheTs) < MODEL_CACHE_TTL_MS) {
    return _modelCache;
  }
  try {
    const resp = await axios.get(`${AI_SERVICE_URL}/api/v1/health`, { timeout: 5_000 });
    const health = resp.data;
    const active    = health.active_model;
    const available = health.models_available ?? {};

    // Use active model if it has weights, else pick first available, else 'cnn'
    let best = 'cnn';
    if (available[active]) {
      best = active;
    } else {
      const fallback = Object.keys(available).find((m) => available[m]);
      if (fallback) best = fallback;
    }

    _modelCache   = best;
    _modelCacheTs = now;
    logger.info(`[CLASSIFY] Best available model resolved to: '${best}'`);
    return best;
  } catch (err) {
    logger.warn(`[CLASSIFY] Could not query AI health, defaulting to 'cnn': ${err.message}`);
    return 'cnn';
  }
}

/**
 * Extract GLCM features via FastAPI and persist them.
 * Called automatically before classification so the Results page
 * always has features — even when the frontend skips POST /api/features.
 */
async function extractAndSaveFeatures(imageId, imagePath) {
  try {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath), {
      filename:    path.basename(imagePath),
      contentType: 'image/jpeg',
    });
    const resp = await axios.post(
      `${AI_SERVICE_URL}/api/v1/glcm`,
      form,
      { headers: form.getHeaders(), timeout: 30_000 }
    );
    const feats = resp.data?.data ?? resp.data;
    const now   = new Date().toISOString();
    const existing = db.prepare('SELECT id FROM features WHERE image_id = ?').get(imageId);
    if (existing) {
      db.prepare(`
        UPDATE features
           SET entropy=?, correlation=?, energy=?, contrast=?,
               mean=?, std_dev=?, variance=?
         WHERE image_id=?
      `).run(
        feats.entropy, feats.correlation, feats.energy,
        feats.contrast, feats.mean, feats.std_dev, feats.variance,
        imageId
      );
    } else {
      const { v4: uuidv4 } = require('uuid');
      db.prepare(`
        INSERT INTO features
          (id, image_id, entropy, correlation, energy, contrast,
           mean, std_dev, variance, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        uuidv4(), imageId,
        feats.entropy, feats.correlation, feats.energy,
        feats.contrast, feats.mean, feats.std_dev, feats.variance, now
      );
    }
    logger.info(`[CLASSIFY] GLCM auto-extracted for ${imageId}`);
  } catch (err) {
    // Non-fatal — log and continue with classification
    logger.warn(`[CLASSIFY] GLCM auto-extraction skipped for ${imageId}: ${err.message}`);
  }
}

router.post('/:imageId', validateImageId, async (req, res, next) => {
  const timer = startTimer();
  try {
    const { imageId } = req.params;
    const rawPath     = req.imageRecord.raw_path;   // absolute path from DB

    // ── Validate the image file exists on disk ────────────────────────────────
    if (!fs.existsSync(rawPath)) {
      return res.status(404).json({
        success: false,
        error: { code: 404, message: `Raw image file not found on disk: ${rawPath}` },
      });
    }

    logger.info(`[CLASSIFY] Starting classification for imageId=${imageId} — ${rawPath}`);

    // ── Auto-extract GLCM features (use enhanced image if preprocessed) ───────
    const processedRow = db
      .prepare('SELECT enhanced_path, resized_path FROM processed_images WHERE image_id = ?')
      .get(imageId);
    const featureImagePath = processedRow?.enhanced_path
      || processedRow?.resized_path
      || rawPath;
    await extractAndSaveFeatures(imageId, featureImagePath);

    // ── Build multipart/form-data for FastAPI ─────────────────────────────────
    const modelName = await getBestAvailableModel();
    logger.info(`[CLASSIFY] Forwarding imageId=${imageId} to AI service using model='${modelName}'`);
    const form = new FormData();
    form.append('image', fs.createReadStream(rawPath), {
      filename:    path.basename(rawPath),
      contentType: 'image/jpeg',
    });
    form.append('model_name',      modelName);
    form.append('generate_gradcam', 'true');

    // ── Call FastAPI POST /api/v1/predict ─────────────────────────────────────
    let aiResponse;
    try {
      aiResponse = await axios.post(
        `${AI_SERVICE_URL}/api/v1/predict`,
        form,
        {
          headers: form.getHeaders(),
          timeout: 120_000,   // model inference can take a moment on CPU
        }
      );
    } catch (axiosErr) {
      const detail =
        axiosErr.response?.data?.detail ||
        axiosErr.response?.data?.error?.message ||
        axiosErr.message;
      logger.error(`[CLASSIFY] AI service error for ${imageId}: ${detail}`);
      return res.status(502).json({
        success: false,
        error: {
          code: 502,
          message: `AI service prediction failed: ${detail}`,
        },
      });
    }

    const aiData = aiResponse.data?.data ?? aiResponse.data;

    // aiData shape from FastAPI /predict:
    // { class, confidence, probabilities, gradcam_path, model_used }
    const predictedClass = aiData.class;
    const confidence     = aiData.confidence;
    const probabilities  = aiData.probabilities ?? {};
    const modelUsed      = aiData.model_used ?? 'unknown';
    const gradcamAbsPath = aiData.gradcam_path ?? null;   // absolute path on AI service

    // ── Convert Grad-CAM absolute path to a static URL ────────────────────────
    // AI service saves to:  <ai-service>/gradcam_output/<image_id>/overlay.png
    // Express serves:       /gradcam/<image_id>/overlay.png   (added in server.js)
    // We need to extract the relative path from within gradcam_output/, NOT just basename.
    let gradcamUrl = null;
    if (gradcamAbsPath) {
      const gradcamOutputMarker = 'gradcam_output' + path.sep;
      const markerIdx = gradcamAbsPath.indexOf(gradcamOutputMarker);
      if (markerIdx !== -1) {
        // Extract path relative to gradcam_output/, replace OS sep with URL sep
        const relPath = gradcamAbsPath
          .substring(markerIdx + gradcamOutputMarker.length)
          .replace(/\\/g, '/');
        gradcamUrl = `/gradcam/${relPath}`;
      } else {
        // Fallback: use just the basename (flat directory structure)
        gradcamUrl = `/gradcam/${path.basename(gradcamAbsPath)}`;
      }
    }

    timer.stop();
    const computationalTimeMs = timer.elapsedMs();
    const now = new Date().toISOString();

    // ── Persist result to database ─────────────────────────────────────────────
    const existing = db
      .prepare('SELECT id FROM results WHERE image_id = ?')
      .get(imageId);

    if (existing) {
      db.prepare(`
        UPDATE results
           SET prediction=?, confidence=?, gradcam_path=?, computational_time=?,
               probabilities=?, model_used=?
         WHERE image_id=?
      `).run(
        predictedClass, confidence, gradcamAbsPath, computationalTimeMs,
        JSON.stringify(probabilities), modelUsed,
        imageId
      );
    } else {
      db.prepare(`
        INSERT INTO results
          (id, image_id, prediction, confidence,
           accuracy, sensitivity, specificity,
           psnr, jaccard, ber, computational_time,
           gradcam_path, probabilities, model_used, created_at)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, ?, ?, ?)
      `).run(
        uuidv4(), imageId, predictedClass, confidence, computationalTimeMs,
        gradcamAbsPath, JSON.stringify(probabilities), modelUsed, now
      );
    }

    logger.info(
      `[CLASSIFY] imageId=${imageId} | class=${predictedClass} | ` +
      `confidence=${confidence} | gradcam=${gradcamUrl} | ${computationalTimeMs.toFixed(1)}ms`
    );

    return res.status(200).json({
      success: true,
      data: {
        image_id:             imageId,
        predicted_class:      predictedClass,
        confidence,
        probabilities,
        gradcam_url:          gradcamUrl,
        model_used:           modelUsed,
        computational_time_ms: parseFloat(computationalTimeMs.toFixed(2)),
        next_step:            `GET /api/results/${imageId}`,
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
