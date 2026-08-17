"""
KLA Denoising / Super-Resolution — V3 Training Script

This is the standalone version of the V3 training portion from
KLA_Final_Clean_Workflow(7).ipynb.

Run from the project root:

    python train.py

Expected project structure:

    train/
        GT/
        NoisyLR/
    data/
        dataset.py
    models/
        model.py
        residual_block.py
    checkpoints/
        best_epoch13.weights.h5

V3 artifacts are written only to:

    checkpoints/v3/
"""

import os

# ---------------------------------------------------------------------
# Reproducibility / TensorFlow configuration
# ---------------------------------------------------------------------
SEED = 42

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

import random
import numpy as np
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------
# Configuration — matches the clean V3 notebook
# ---------------------------------------------------------------------
BATCH_SIZE = 16
V3_EPOCHS = 25
AUGMENT_FRACTION = 0.34
V3_LR = 1e-4

# Conservative V3 acceptance thresholds used later for model selection.
MIN_PSNR_IMPROVEMENT = 0.10
MIN_SSIM_IMPROVEMENT = 0.003

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# train.py is expected to live in the project root.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

GT_DIR = os.path.join(PROJECT_DIR, "train", "GT")
NOISY_DIR = os.path.join(PROJECT_DIR, "train", "NoisyLR")

V1_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_epoch13.weights.h5",
)

V3_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "v3",
)

# ---------------------------------------------------------------------
# Imports from the existing project
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split

from data.dataset import KLADataset
from models.model import model as architecture


def verify_environment():
    """Verify the required project files and GPU before training."""
    required = [
        GT_DIR,
        NOISY_DIR,
        os.path.join(PROJECT_DIR, "data", "dataset.py"),
        os.path.join(PROJECT_DIR, "models", "model.py"),
        V1_CHECKPOINT,
    ]

    missing = [path for path in required if not os.path.exists(path)]

    if missing:
        raise FileNotFoundError(
            "Missing required project paths:\n"
            + "\n".join(missing)
        )

    gpus = tf.config.list_physical_devices("GPU")

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("Python:", __import__("sys").version.split()[0])
    print("Project:", PROJECT_DIR)
    print("GPUs:", gpus)

    if not gpus:
        raise RuntimeError(
            "STOP: No GPU detected. V3 training is blocked to prevent "
            "accidental CPU training."
        )

    for gpu in gpus:
        print("GPU:", gpu)

    print()


def load_data():
    """Load the original real NoisyLR/GT paired dataset into RAM."""
    print("=" * 60)
    print("LOADING ORIGINAL KLA DATA")
    print("=" * 60)

    dataset = KLADataset(
        GT_DIR,
        NOISY_DIR,
    )

    X, Y = dataset.load()

    print("X:", X.shape, X.dtype)
    print("Y:", Y.shape, Y.dtype)
    print("X range:", float(X.min()), "to", float(X.max()))
    print("Y range:", float(Y.min()), "to", float(Y.max()))
    print()

    return X, Y


def make_validation_split(X, Y):
    """Preserve the exact original 90/10 validation protocol."""
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=0.1,
        random_state=42,
    )

    print("=" * 60)
    print("EXACT ORIGINAL VALIDATION SPLIT")
    print("=" * 60)
    print("Training:")
    print("  X_train:", X_train.shape)
    print("  Y_train:", Y_train.shape)
    print()
    print("Validation:")
    print("  X_val:", X_val.shape)
    print("  Y_val:", Y_val.shape)
    print()

    return X_train, X_val, Y_train, Y_val


def add_mild_noise(image, rng):
    """
    Apply the exact mild synthetic degradation used by V3.

    The image is already a real NoisyLR sample. Synthetic degradation
    is added only to selected training samples.
    """
    image = image.astype(np.float32, copy=True)

    noise_type = rng.choice(
        ["gaussian", "speckle"]
    )

    if noise_type == "gaussian":
        sigma = rng.uniform(
            0.005,
            0.02,
        )

        noise = rng.normal(
            0.0,
            sigma,
            size=image.shape,
        ).astype(np.float32)

        out = image + noise

    else:
        strength = rng.uniform(
            0.01,
            0.05,
        )

        noise = rng.normal(
            0.0,
            strength,
            size=image.shape,
        ).astype(np.float32)

        out = image + image * noise

    return np.clip(
        out,
        0.0,
        1.0,
    ).astype(np.float32)


def build_v3_training_data(X_train, X_val, Y_train, Y_val):
    """
    Build the V3 in-memory tf.data pipeline.

    34% of training samples receive mild synthetic degradation.
    Validation remains completely untouched.
    """
    print("=" * 60)
    print("BUILDING V3 TRAINING DATA")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    X_train_v3 = X_train.copy()

    n_aug = int(
        len(X_train_v3) * AUGMENT_FRACTION
    )

    aug_indices = rng.choice(
        len(X_train_v3),
        size=n_aug,
        replace=False,
    )

    for idx in aug_indices:
        X_train_v3[idx] = add_mild_noise(
            X_train_v3[idx],
            rng,
        )

    print("Original training images:", len(X_train))
    print("Mildly augmented images :", n_aug)
    print(
        "Unchanged images        :",
        len(X_train) - n_aug,
    )
    print("Validation images       :", len(X_val))
    print()

    train_ds_v3 = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_train_v3, Y_train)
        )
        .shuffle(
            buffer_size=len(X_train_v3),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds_v3 = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_val, Y_val)
        )
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print("V3 datasets ready.")
    print(
        "Training batches  :",
        tf.data.experimental.cardinality(train_ds_v3).numpy(),
    )
    print(
        "Validation batches:",
        tf.data.experimental.cardinality(val_ds_v3).numpy(),
    )
    print()

    return train_ds_v3, val_ds_v3


def calculate_metrics(model, x, y, batch_size=16):
    """Calculate PSNR and SSIM on the untouched validation set."""
    pred = model.predict(
        x,
        batch_size=batch_size,
        verbose=1,
    )

    pred = np.clip(
        pred,
        0.0,
        1.0,
    ).astype(np.float32)

    psnr = float(
        tf.reduce_mean(
            tf.image.psnr(
                y,
                pred,
                max_val=1.0,
            )
        )
    )

    ssim = float(
        tf.reduce_mean(
            tf.image.ssim(
                y,
                pred,
                max_val=1.0,
            )
        )
    )

    return psnr, ssim, pred


def build_v1_and_v3_models():
    """
    Load the locked V1 checkpoint and create an independent V3 model
    with identical architecture and initial V1 weights.
    """
    print("=" * 60)
    print("LOADING V1 + INITIALIZING V3")
    print("=" * 60)

    # V1 is the existing architecture object with the locked weights.
    v1_model = architecture

    v1_model.load_weights(
        V1_CHECKPOINT
    )

    print("V1 weights loaded successfully.")

    # Independent model object with exactly the same architecture.
    v3_model = tf.keras.models.clone_model(
        architecture
    )

    v3_model.set_weights(
        v1_model.get_weights()
    )

    same = all(
        np.array_equal(
            w1.numpy(),
            w3.numpy(),
        )
        for w1, w3 in zip(
            v1_model.weights,
            v3_model.weights,
        )
    )

    print("V3 initialized from V1 weights.")
    print(
        "V1 and V3 initially identical:",
        same,
    )
    print()

    if not same:
        raise RuntimeError(
            "V3 initialization check failed: V1 and V3 weights "
            "are not identical."
        )

    return v1_model, v3_model


def configure_callbacks():
    """Create the V3 callbacks exactly under checkpoints/v3/."""
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        CSVLogger,
    )

    os.makedirs(
        V3_DIR,
        exist_ok=True,
    )

    callbacks_v3 = [
        ModelCheckpoint(
            os.path.join(
                V3_DIR,
                "best.keras",
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        ModelCheckpoint(
            os.path.join(
                V3_DIR,
                "best.weights.h5",
            ),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        CSVLogger(
            os.path.join(
                V3_DIR,
                "training_log.csv",
            ),
            append=False,
        ),
    ]

    return callbacks_v3


def train_v3(v3_model, train_ds_v3, val_ds_v3):
    """Fine-tune V3 from V1 weights."""
    print("=" * 60)
    print("V3 TRAINING SETUP")
    print("=" * 60)

    # Preserve the existing model loss definition from the architecture.
    # The project model defines MAE.
    v3_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=V3_LR,
        ),
        loss="mae",
    )

    print("Learning rate:", V3_LR)
    print("Loss:", v3_model.loss)
    print("Epochs:", V3_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print()

    callbacks_v3 = configure_callbacks()

    print("=" * 60)
    print("V3 TRAINING")
    print("=" * 60)

    history_v3 = v3_model.fit(
        train_ds_v3,
        validation_data=val_ds_v3,
        epochs=V3_EPOCHS,
        callbacks=callbacks_v3,
        verbose=1,
    )

    # Save final V3 artifact only under checkpoints/v3/.
    v3_model.save(
        os.path.join(
            V3_DIR,
            "final.keras",
        )
    )

    best_v3_loss = min(
        history_v3.history["val_loss"]
    )

    print()
    print("=" * 60)
    print("V3 TRAINING COMPLETE")
    print("=" * 60)
    print("Best V3 val_loss:", best_v3_loss)
    print(
        "V3 artifacts:",
        V3_DIR,
    )
    print()

    return history_v3


def main():
    print("=" * 60)
    print("KLA — RESIDUALSRCNN V3 TRAINING")
    print("=" * 60)
    print()
    print("Configuration:")
    print("  BATCH_SIZE       :", BATCH_SIZE)
    print("  V3_EPOCHS        :", V3_EPOCHS)
    print("  AUGMENT_FRACTION :", AUGMENT_FRACTION)
    print("  V3_LR            :", V3_LR)
    print("  SEED             :", SEED)
    print()

    verify_environment()

    # Load real paired KLA data.
    X, Y = load_data()

    # Preserve exact original validation protocol.
    X_train, X_val, Y_train, Y_val = make_validation_split(
        X,
        Y,
    )

    # Load locked V1 and initialize independent V3 from it.
    v1_model, v3_model = build_v1_and_v3_models()

    # Establish V1 baseline on untouched validation.
    print("=" * 60)
    print("V1 BASELINE — ORIGINAL VALIDATION")
    print("=" * 60)

    v1_psnr, v1_ssim, _ = calculate_metrics(
        v1_model,
        X_val,
        Y_val,
        BATCH_SIZE,
    )

    print()
    print("V1 PSNR:", v1_psnr)
    print("V1 SSIM:", v1_ssim)
    print()

    # Build V3 training data. Validation is never modified.
    train_ds_v3, val_ds_v3 = build_v3_training_data(
        X_train,
        X_val,
        Y_train,
        Y_val,
    )

    # Fine-tune V3.
    history_v3 = train_v3(
        v3_model,
        train_ds_v3,
        val_ds_v3,
    )

    # Evaluate V3 on the SAME untouched validation set.
    print("=" * 60)
    print("V1 vs V3 — ORIGINAL VALIDATION")
    print("=" * 60)

    v3_psnr, v3_ssim, _ = calculate_metrics(
        v3_model,
        X_val,
        Y_val,
        BATCH_SIZE,
    )

    psnr_gain = v3_psnr - v1_psnr
    ssim_gain = v3_ssim - v1_ssim

    print()
    print("V1")
    print("  PSNR:", v1_psnr)
    print("  SSIM:", v1_ssim)

    print()
    print("V3")
    print("  PSNR:", v3_psnr)
    print("  SSIM:", v3_ssim)

    print()
    print("Improvement")
    print("  PSNR:", f"{psnr_gain:+.6f}")
    print("  SSIM:", f"{ssim_gain:+.6f}")

    v3_wins = (
        psnr_gain >= MIN_PSNR_IMPROVEMENT
        and ssim_gain >= MIN_SSIM_IMPROVEMENT
    )

    print()
    print("V3 PSNR threshold:", MIN_PSNR_IMPROVEMENT)
    print("V3 SSIM threshold:", MIN_SSIM_IMPROVEMENT)

    if v3_wins:
        print("V3 meets both improvement thresholds.")
    else:
        print("V3 does NOT meet both improvement thresholds.")
        print("V1 remains the conservative final-model candidate.")

    # Keep a small machine-readable summary alongside the training log.
    import json

    results = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "v3_epochs": V3_EPOCHS,
        "augment_fraction": AUGMENT_FRACTION,
        "v3_learning_rate": V3_LR,
        "v1_psnr": v1_psnr,
        "v1_ssim": v1_ssim,
        "v3_psnr": v3_psnr,
        "v3_ssim": v3_ssim,
        "psnr_gain": psnr_gain,
        "ssim_gain": ssim_gain,
        "min_psnr_improvement": MIN_PSNR_IMPROVEMENT,
        "min_ssim_improvement": MIN_SSIM_IMPROVEMENT,
        "v3_meets_thresholds": v3_wins,
        "best_v3_val_loss": float(
            min(history_v3.history["val_loss"])
        ),
    }

    with open(
        os.path.join(
            V3_DIR,
            "validation_results.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("TRAINING SCRIPT FINISHED")
    print("=" * 60)
    print("V1 checkpoint was not modified.")
    print("All V3 artifacts are under:")
    print(" ", V3_DIR)


if __name__ == "__main__":
    main()
