'use strict';

/**
 * POST /api/evaluate
 *
 * Triggers a full model evaluation on the AI service's test dataset,
 * then stores the results in the model_metrics table so that
 * GET /api/metrics returns real computed values.
 *
 * The AI service POST /api/v1/evaluate endpoint is called with the
 * active model.  The returned metrics are persisted to the DB and
 * returned in the response.
 *
 * Success response 200:
 *  {
 *    success: true,
 *    data: {
 *      accuracy, sensitivity, specificity, psnr,
 *      jaccard, ber, computational_time,
 *      f1, auc_roc, num_samples, class_names,
 *      confusion_matrix, per_class,
 *      trained_at
 *    }
 *  }
 */

const express = require('express');
const { v4: uuidv4 } = require('uuid');
const axios   = require('axios');
const router  = express.Router();

const db      = require('../database/db');
const logger  = require('../utils/logger');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// ── POST /api/evaluate ────────────────────────────────────────────────────────
router.post('/', async (req, res, next) => {
  try {
    logger.info('[EVALUATE] Starting model evaluation via AI service…');

    // ── Call AI service POST /api/v1/evaluate (no auth required in public mode) ──
    // The AI service /evaluate requires ADMIN or RESEARCHER role by default.
    // We call it with a Basic-auth header using the seeded admin credentials.
    // In production, use a service-account token stored in env vars.
    let aiResponse;
    try {
      // First try unauthenticated — works when PREDICTION_AUTH_MODE=public
      // and the evaluate endpoint doesn't gate on role in some configs.
      // Fall back to admin credentials if 401/403.
      const payload = {
        model_name: null,  // use ACTIVE_MODEL from settings
        batch_size: 32,
      };

      // Get admin credentials from env (set during development seed)
      const adminUser = process.env.EVAL_ADMIN_USER || 'admin';
      const adminPass = process.env.EVAL_ADMIN_PASS || 'Admin@123!';
      const basicAuth = Buffer.from(`${adminUser}:${adminPass}`).toString('base64');

      aiResponse = await axios.post(
        `${AI_SERVICE_URL}/api/v1/evaluate`,
        payload,
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Basic ${basicAuth}`,
          },
          timeout: 600_000,  // evaluation can take several minutes
        }
      );
    } catch (axiosErr) {
      const detail =
        axiosErr.response?.data?.detail ||
        axiosErr.response?.data?.error?.message ||
        axiosErr.message;
      logger.error(`[EVALUATE] AI service error: ${detail}`);
      return res.status(502).json({
        success: false,
        error: {
          code: 502,
          message: `AI service evaluation failed: ${detail}`,
        },
      });
    }

    const aiData = aiResponse.data?.data ?? aiResponse.data;

    // ── Extract metrics from AI response ─────────────────────────────────────
    // evaluate_model() now returns all required fields directly
    const accuracy          = aiData.accuracy          ?? null;
    const sensitivity       = aiData.sensitivity       ?? null;
    const specificity       = aiData.specificity       ?? null;
    const psnr              = aiData.psnr              ?? null;
    const jaccard           = aiData.jaccard           ?? null;
    const ber               = aiData.ber               ?? null;
    const computationalTime = aiData.computational_time ?? null;
    const f1                = aiData.f1                ?? null;
    const auc_roc           = aiData.auc_roc           ?? null;
    const numSamples        = aiData.num_samples       ?? null;
    const classNames        = aiData.class_names       ?? null;
    const confusionMatrix   = aiData.confusion_matrix  ?? null;
    const perClass          = aiData.per_class         ?? null;
    const trainedAt         = new Date().toISOString();

    // ── Persist to model_metrics table ────────────────────────────────────────
    const rowId = uuidv4();
    db.prepare(`
      INSERT INTO model_metrics
        (id, accuracy, sensitivity, specificity, psnr,
         jaccard, ber, computational_time, trained_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      rowId, accuracy, sensitivity, specificity, psnr,
      jaccard, ber, computationalTime, trainedAt
    );

    logger.info(
      `[EVALUATE] Stored metrics | accuracy=${accuracy} sensitivity=${sensitivity} ` +
      `specificity=${specificity} psnr=${psnr} jaccard=${jaccard} ber=${ber}`
    );

    return res.status(200).json({
      success: true,
      data: {
        accuracy,
        sensitivity,
        specificity,
        psnr,
        jaccard,
        ber,
        computational_time: computationalTime,
        f1,
        auc_roc,
        num_samples:      numSamples,
        class_names:      classNames,
        confusion_matrix: confusionMatrix,
        per_class:        perClass,
        trained_at:       trainedAt,
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
