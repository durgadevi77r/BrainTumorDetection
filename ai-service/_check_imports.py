"""Quick import verification for all training pipeline modules."""
import sys

checks = []

try:
    from app.core.config import settings
    checks.append("app.core.config: OK")
except Exception as e:
    checks.append(f"app.core.config: FAIL — {e}")

try:
    from training.config import TrainingConfig, DEFAULT_TRAINING_CONFIG
    checks.append("training.config: OK")
except Exception as e:
    checks.append(f"training.config: FAIL — {e}")

try:
    from training.callbacks import build_callbacks, CallbackBundle, get_best_checkpoint_path
    checks.append("training.callbacks: OK")
except Exception as e:
    checks.append(f"training.callbacks: FAIL — {e}")

try:
    from training.checkpoints import save_checkpoint_info, load_best_weights, list_checkpoints
    checks.append("training.checkpoints: OK")
except Exception as e:
    checks.append(f"training.checkpoints: FAIL — {e}")

try:
    from training.experiment import Experiment, ExperimentRegistry
    checks.append("training.experiment: OK")
except Exception as e:
    checks.append(f"training.experiment: FAIL — {e}")

try:
    from app.models.architectures import build_model, build_optimizer, unfreeze_top_layers
    checks.append("app.models.architectures: OK")
except Exception as e:
    checks.append(f"app.models.architectures: FAIL — {e}")

try:
    from app.models.save_model import save_keras_model, save_best_checkpoint, load_best_checkpoint
    checks.append("app.models.save_model: OK")
except Exception as e:
    checks.append(f"app.models.save_model: FAIL — {e}")

try:
    from app.models.train import train_model, _run_epoch
    checks.append("app.models.train: OK")
except Exception as e:
    checks.append(f"app.models.train: FAIL — {e}")

try:
    from training.trainer import Trainer, train, _build_arg_parser
    checks.append("training.trainer: OK")
except Exception as e:
    checks.append(f"training.trainer: FAIL — {e}")

try:
    from app.models.evaluate import evaluate_model
    checks.append("app.models.evaluate: OK")
except Exception as e:
    checks.append(f"app.models.evaluate: FAIL — {e}")

try:
    from app.preprocessing.preprocess import build_generators, preprocess_for_inference
    checks.append("app.preprocessing.preprocess: OK")
except Exception as e:
    checks.append(f"app.preprocessing.preprocess: FAIL — {e}")

for c in checks:
    print(c)

fails = [c for c in checks if "FAIL" in c]
print(f"\n{'='*50}")
print(f"{'ALL OK' if not fails else str(len(fails)) + ' FAILURES'}")
sys.exit(1 if fails else 0)
