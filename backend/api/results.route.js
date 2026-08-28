'use strict';

const express = require('express');
const path    = require('path');
const router  = express.Router();

const db      = require('../database/db');
const config  = require('../config');
const { validateImageId } = require('../middleware/validateRequest');
const { toStaticUrl }     = require('../utils/imageUtils');
const logger  = require('../utils/logger');

/**
 * Convert an absolute filesystem path to an Express static URL.
 * Returns null when the path is falsy.
 *
 * Examples:
 *  /abs/path/dataset/processed/resized/foo.png → /processed/resized/foo.png
 *  /abs/path/uploads/foo.jpg                   → /uploads/foo.jpg
 */
function toUrl(absPath, baseDir, prefix) {
  if (!absPath) return null;
  try {
    return prefix + toStaticUrl(absPath, baseDir);
  } catch (_) {
    return null;
  }
}

/**
 * GET /api/results/:imageId
 *
 * Returns the complete pipeline result for a given image by joining
 * all four database tables: images, processed_images, features, results.
 *
 * All filesystem paths are converted to static HTTP URLs for the frontend.
 *
 * Success response 200:
 *  {
 *    success: true,
 *    data: {
 *      image_id, filename, upload_time,
 *      pipeline_complete: true,
 *      paths:       { raw_url, resized_url, enhanced_url, denoised_url, segmented_url },
 *      features:    { entropy, correlation, energy, contrast, mean, std_dev, variance },
 *      result:      { predicted_class, confidence, probabilities, gradcam_url, model_used },
 *      metrics:     { accuracy, sensitivity, specificity, psnr, jaccard, ber, computational_time }
 *    }
 *  }
 *
 * 202 is returned when the pipeline has not yet completed for this image.
 */
router.get('/:imageId', validateImageId, (req, res, next) => {
  try {
    const { imageId } = req.params;

    // ── Fetch processed paths ─────────────────────────────────────────────────
    const processed = db
      .prepare('SELECT * FROM processed_images WHERE image_id = ?')
      .get(imageId);

    // ── Fetch features ────────────────────────────────────────────────────────
    const featureRow = db
      .prepare('SELECT * FROM features WHERE image_id = ?')
      .get(imageId);

    // ── Fetch classification result ───────────────────────────────────────────
    const resultRow = db
      .prepare('SELECT * FROM results WHERE image_id = ?')
      .get(imageId);

    // ── Build URL helpers ─────────────────────────────────────────────────────
    // Express static mounts:
    //   /uploads   → config.paths.uploadDir
    //   /processed → config.paths.processedDir   (= dataset/processed)
    //   /gradcam   → <project>/ai-service/gradcam_output

    const uploadBase   = config.paths.uploadDir;
    const processedBase = config.paths.processedDir;

    const rawUrl = req.imageRecord.raw_path
      ? '/uploads/' + path.basename(req.imageRecord.raw_path)
      : null;

    const resizedUrl   = processed?.resized_path
      ? toUrl(processed.resized_path,   processedBase, '/processed')
      : null;
    const enhancedUrl  = processed?.enhanced_path
      ? toUrl(processed.enhanced_path,  processedBase, '/processed')
      : null;
    const denoisedUrl  = processed?.denoised_path
      ? toUrl(processed.denoised_path,  processedBase, '/processed')
      : null;
    const segmentedUrl = processed?.segmented_path
      ? toUrl(processed.segmented_path, processedBase, '/processed')
      : null;

    // Grad-CAM: stored as absolute path in <ai-service>/gradcam_output/<image_id>/overlay.png
    // Express serves /gradcam/ → <ai-service>/gradcam_output/
    // Extract the relative path from within gradcam_output/ for nested subdirectory structure.
    let gradcamUrl = null;
    if (resultRow?.gradcam_path) {
      const sep = resultRow.gradcam_path.includes('/') ? '/' : path.sep;
      const marker = `gradcam_output${sep}`;
      const markerIdx = resultRow.gradcam_path.indexOf(marker);
      if (markerIdx !== -1) {
        const relPath = resultRow.gradcam_path
          .substring(markerIdx + marker.length)
          .replace(/\\/g, '/');
        gradcamUrl = `/gradcam/${relPath}`;
      } else {
        gradcamUrl = `/gradcam/${path.basename(resultRow.gradcam_path)}`;
      }
    }

    // ── Parse stored probabilities JSON ───────────────────────────────────────
    let probabilities = null;
    if (resultRow?.probabilities) {
      try {
        probabilities = JSON.parse(resultRow.probabilities);
      } catch (_) {
        probabilities = null;
      }
    }

    // ── Pipeline completion check ─────────────────────────────────────────────
    const pipelineComplete = !!(processed && resultRow);

    if (!pipelineComplete) {
      return res.status(202).json({
        success: true,
        data: {
          image_id:   imageId,
          filename:   req.imageRecord.filename,
          upload_time: req.imageRecord.upload_time,
          pipeline_complete: false,
          completed_steps: {
            upload:     true,
            preprocess: !!processed,
            classify:   !!resultRow,
          },
          paths: {
            raw_url:       rawUrl,
            resized_url:   resizedUrl,
            enhanced_url:  enhancedUrl,
            denoised_url:  denoisedUrl,
            segmented_url: segmentedUrl,
          },
          features: featureRow ? {
            entropy:     featureRow.entropy,
            correlation: featureRow.correlation,
            energy:      featureRow.energy,
            contrast:    featureRow.contrast,
            mean:        featureRow.mean,
            std_dev:     featureRow.std_dev,
            variance:    featureRow.variance,
          } : null,
          result:  null,
          metrics: null,
        },
      });
    }

    logger.info(`[RESULTS] Full result for imageId=${imageId}`);

    return res.status(200).json({
      success: true,
      data: {
        image_id:    imageId,
        filename:    req.imageRecord.filename,
        upload_time: req.imageRecord.upload_time,
        pipeline_complete: true,
        paths: {
          raw_url:       rawUrl,
          resized_url:   resizedUrl,
          enhanced_url:  enhancedUrl,
          denoised_url:  denoisedUrl,
          segmented_url: segmentedUrl,
        },
        features: featureRow ? {
          entropy:     featureRow.entropy,
          correlation: featureRow.correlation,
          energy:      featureRow.energy,
          contrast:    featureRow.contrast,
          mean:        featureRow.mean,
          std_dev:     featureRow.std_dev,
          variance:    featureRow.variance,
        } : null,
        result: {
          predicted_class: resultRow.prediction,
          confidence:      resultRow.confidence,
          probabilities,
          gradcam_url:     gradcamUrl,
          model_used:      resultRow.model_used ?? null,
        },
        metrics: {
          accuracy:           resultRow.accuracy,
          sensitivity:        resultRow.sensitivity,
          specificity:        resultRow.specificity,
          psnr:               resultRow.psnr,
          jaccard:            resultRow.jaccard,
          ber:                resultRow.ber,
          computational_time: resultRow.computational_time,
        },
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
