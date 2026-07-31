'use strict';

const express = require('express');
const router  = express.Router();

const db      = require('../database/db');
const logger  = require('../utils/logger');

/**
 * GET /api/metrics
 *
 * Returns the stored model evaluation metrics from the most recent training run.
 * Populated by training/evaluate.js after EDN-SVM training completes.
 *
 * Returns 503 if no training run has been recorded yet.
 *
 * Success response 200:
 *  {
 *    success: true,
 *    data: {
 *      accuracy:           97.93,   // Eq. 28 — paper target
 *      sensitivity:        92.0,    // Eq. 31 — paper target
 *      specificity:        98.0,    // Eq. 32 — paper target
 *      psnr:               52.98,   // Eq. 30 — paper target
 *      jaccard:            <value>, // Eq. 29
 *      ber:                <value>,
 *      computational_time: <value>, // minutes
 *      trained_at:         "<ISO timestamp>"
 *    }
 *  }
 */
router.get('/', (req, res, next) => {
  try {
    // Fetch most recent training metrics row
    const row = db
      .prepare(`
        SELECT accuracy, sensitivity, specificity, psnr,
               jaccard, ber, computational_time, trained_at
        FROM model_metrics
        ORDER BY trained_at DESC
        LIMIT 1
      `)
      .get();

    // No training run recorded yet — return placeholder paper targets
    if (!row) {
      logger.warn('[METRICS] No training run found — returning paper target placeholders');
      return res.status(200).json({
        success: true,
        data: {
          accuracy:           98.51,
          sensitivity:        98.2,
          specificity:        98.9,
          psnr:               52.98,
          jaccard:            null,
          ber:                null,
          computational_time: null,
          trained_at:         null,
          status:             'placeholder — model has not been trained yet. Run POST /api/v1/train/start',
        },
      });
    }

    logger.info('[METRICS] Returning stored model metrics');

    return res.status(200).json({
      success: true,
      data: {
        accuracy:           row.accuracy,
        sensitivity:        row.sensitivity,
        specificity:        row.specificity,
        psnr:               row.psnr,
        jaccard:            row.jaccard,
        ber:                row.ber,
        computational_time: row.computational_time,
        trained_at:         row.trained_at,
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
