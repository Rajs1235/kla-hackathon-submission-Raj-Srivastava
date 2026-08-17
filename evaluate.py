#!/usr/bin/env python3
"""
KLA Image Restoration - Standalone Evaluation Script

Usage:
    python eval.py <test_directory> <output_directory>

Example:
    python eval.py ./test ./submission/predictions

The script loads the final V3 weights, runs batched inference over
all .npy test images, writes predictions using the original filenames,
and validates the generated outputs.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "checkpoints"/ "best.weights.h5"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run KLA ResidualSRCNN restoration inference."
    )
    parser.add_argument(
        "test_directory",
        type=Path,
        help="Directory containing test .npy images."
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory for restored .npy images."
    )
    return parser.parse_args()


def load_final_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Final V3 weights not found: {WEIGHTS_PATH}"
        )

    from models.model import model as architecture

    model = tf.keras.models.clone_model(architecture)
    model.load_weights(WEIGHTS_PATH)

    return model


def get_test_files(test_directory):
    if not test_directory.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {test_directory}"
        )

    files = sorted(
        p for p in test_directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".npy"
    )

    if not files:
        raise RuntimeError(
            f"No .npy files found in {test_directory}"
        )

    return files


def load_input(path):
    image = np.load(path).astype(np.float32)

    if image.shape != (128, 128):
        raise ValueError(
            f"{path.name}: expected (128, 128), got {image.shape}"
        )

    image = np.clip(image, 0.0, 1.0)

    # (128, 128) -> (128, 128, 1)
    return image[..., np.newaxis]


def run_inference(model, test_files, output_directory, batch_size=32):
    output_directory.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    for start_index in range(0, len(test_files), batch_size):
        batch_paths = test_files[
            start_index:start_index + batch_size
        ]

        batch = np.stack(
            [load_input(path) for path in batch_paths],
            axis=0
        )

        predictions = model.predict(
            batch,
            verbose=0
        )

        expected_shape = (
            len(batch_paths),
            256,
            256,
            1
        )

        if predictions.shape != expected_shape:
            raise ValueError(
                f"Unexpected prediction shape: {predictions.shape}; "
                f"expected {expected_shape}"
            )

        predictions = np.clip(
            predictions,
            0.0,
            1.0
        ).astype(np.float32)

        for path, prediction in zip(batch_paths, predictions):
            np.save(
                output_directory / path.name,
                prediction
            )

        completed = start_index + len(batch_paths)
        print(
            f"{completed}/{len(test_files)} completed",
            flush=True
        )

    return time.perf_counter() - start


def validate_outputs(test_files, output_directory):
    output_files = sorted(
        p for p in output_directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".npy"
    )

    test_names = {p.name for p in test_files}
    output_names = {p.name for p in output_files}

    missing = sorted(test_names - output_names)
    extra = sorted(output_names - test_names)

    bad_shapes = []
    bad_values = []

    for path in output_files:
        array = np.load(path)

        if array.shape != (256, 256, 1):
            bad_shapes.append(
                (path.name, array.shape)
            )

        if not np.all(np.isfinite(array)):
            bad_values.append(path.name)
            continue

        if array.min() < 0.0 or array.max() > 1.0:
            bad_values.append(path.name)

    print("\n========================================")
    print("FINAL SUBMISSION VALIDATION")
    print("========================================")
    print("Test files      :", len(test_files))
    print("Output files    :", len(output_files))
    print("Missing         :", len(missing))
    print("Extra           :", len(extra))
    print("Bad shapes      :", len(bad_shapes))
    print("NaN / Inf       :", len(bad_values))

    if missing:
        print("\nMissing:")
        for name in missing:
            print(" ", name)

    if extra:
        print("\nExtra:")
        for name in extra:
            print(" ", name)

    if bad_shapes:
        print("\nBad shapes:")
        for name, shape in bad_shapes[:10]:
            print(" ", name, shape)

    if bad_values:
        print("\nBad values:")
        for name in bad_values[:10]:
            print(" ", name)

    if (
        missing
        or extra
        or bad_shapes
        or bad_values
        or len(test_files) != len(output_files)
    ):
        raise RuntimeError(
            "Submission validation failed."
        )

    print("\nALL SUBMISSION CHECKS PASSED.")


def main():
    args = parse_args()

    test_directory = args.test_directory.resolve()
    output_directory = args.output_directory.resolve()

    print("========================================")
    print("KLA FINAL EVALUATION")
    print("========================================")
    print("Project root :", PROJECT_ROOT)
    print("Test         :", test_directory)
    print("Output       :", output_directory)
    print("Weights      :", WEIGHTS_PATH)

    print("\nLoading final V3 model...")
    model = load_final_model()

    print("Model :", model.name)
    print("Input :", model.input_shape)
    print("Output:", model.output_shape)

    test_files = get_test_files(test_directory)
    print("\nTest images:", len(test_files))

    elapsed = run_inference(
        model,
        test_files,
        output_directory,
        batch_size=32
    )

    print("\n========================================")
    print("INFERENCE COMPLETE")
    print("========================================")
    print("Images        :", len(test_files))
    print(f"Total time    : {elapsed / 60:.2f} minutes")
    print(
        "Average/image : "
        f"{elapsed / len(test_files):.4f} seconds"
    )

    validate_outputs(
        test_files,
        output_directory
    )


if __name__ == "__main__":
    main()
