# KLA AI Hackathon --- Image Restoration

## AI-Based Restoration of Degraded Images

This repository contains the reproducible training and inference
pipeline for the KLA AI Hackathon image-restoration challenge.

The objective is to restore signal-degraded images affected by noise and
spatial-resolution loss, reconstructing the original high-resolution
image as accurately as possible.

The official challenge evaluates restoration quality using **SSIM,
pSNR/PSNR, and LPIPS**, together with **end-to-end inference time on an
NVIDIA H100 GPU**.

------------------------------------------------------------------------

## 1. Problem

The challenge provides paired training data consisting of:

-   **GT** --- high-resolution ground-truth images
-   **NoisyLR** --- degraded, lower-resolution images

The documented degradation scenarios include:

-   Speckle noise
-   Gaussian noise
-   Spatial downsampling
    -   `512 × 512 → 256 × 256`
    -   `256 × 256 → 128 × 128`

The test set contains both in-distribution and out-of-distribution
samples, so robustness and generalization are important.

------------------------------------------------------------------------

## 2. Solution

The submitted solution uses a residual CNN super-resolution/restoration
model named:

**ResidualSRCNN**

The current architecture uses:

-   Initial convolution
-   Batch normalization
-   ReLU activation
-   Multiple residual blocks
-   64 feature channels
-   Residual skip connections
-   2× bilinear upsampling
-   Final reconstruction layer

The model is designed for the current `128 × 128 × 1 → 256 × 256 × 1`
restoration path used by the training pipeline.

The architecture is defined in:

``` text
models/model.py
```

------------------------------------------------------------------------

## 3. Training Strategy

The final training pipeline is based on the clean V3 workflow.

### Data

Real degraded images from:

``` text
train/NoisyLR/
```

are used as the primary model inputs, paired with:

``` text
train/GT/
```

as targets.

### Validation

The validation split is kept reproducible using:

``` text
random_state = 42
```

Validation data is not synthetically degraded or augmented.

### V3 fine-tuning

The V3 training configuration uses:

``` text
Batch size:        16
Epochs:            25
Learning rate:     1e-4
Augmentation:      34% of training samples
Random seed:       42
```

The V3 model is initialized from the locked V1 checkpoint:

``` text
checkpoints/best_epoch13.weights.h5
```

The V1 checkpoint is preserved and is not overwritten.

------------------------------------------------------------------------

## 4. Model Selection

V3 is not considered better merely because its training loss decreases.

The trained V3 model is compared against the locked V1 baseline on the
same validation split using restoration metrics such as:

-   PSNR/pSNR
-   SSIM
-   LPIPS

The final submitted checkpoint is selected only after this comparison.

The final submission model will be stored as:

``` text
checkpoints/final.weights.h5
```

This file represents the exact model used to generate the submitted test
outputs.

------------------------------------------------------------------------

## 5. Repository Structure

The final submission package is organized as:

``` text
KLA_Final_Submission/
│
├── README.md
├── requirements.txt
│
├── train.py
├── evaluate.py
│
├── models/
│   ├── model.py
│   └── residual_block.py
│
├── checkpoints/
│   └── final.weights.h5
│
└── outputs/
    └── denoised/
```

### File responsibilities

  File / Directory                 Purpose
  -------------------------------- --------------------------------------------------------
  `train.py`                       Reproduces training/fine-tuning of the submitted model
  `evaluate.py`                    Standalone command-line inference script
  `models/`                        Model architecture and residual-block implementation
  `checkpoints/final.weights.h5`   Final submitted model weights
  `outputs/denoised/`              Denoised outputs generated from the test set
  `requirements.txt`               Complete Python environment specification
  `README.md`                      Reproduction and usage documentation

------------------------------------------------------------------------

## 6. Training

Run training from the project root:

``` bash
python train.py
```

Training requires a CUDA-compatible NVIDIA GPU for practical execution.

The training pipeline should:

1.  Load paired `NoisyLR` and GT data.
2.  Reproduce the validation split using `random_state=42`.
3.  Apply mild synthetic degradation to the configured fraction of
    training samples.
4.  Initialize the model from the locked V1 checkpoint.
5.  Fine-tune using the V3 configuration.
6.  Save V3 checkpoints and training history.
7.  Evaluate V3 against the V1 baseline.
8.  Select the final checkpoint after validation comparison.

------------------------------------------------------------------------

## 7. Evaluation / Test Inference

The official challenge requires a standalone Python evaluation script.

The intended interface is:

``` bash
python evaluate.py --input_dir <TEST_DIRECTORY> --output_dir <OUTPUT_DIRECTORY>
```

Example:

``` bash
python evaluate.py \
    --input_dir test \
    --output_dir outputs/denoised
```

The evaluation script is intended to:

1.  Accept the test-image directory from the command line.
2.  Accept the output directory from the command line.
3.  Construct the submitted model.
4.  Load `checkpoints/final.weights.h5`.
5.  Run inference over all test inputs.
6.  Write the restored images to the requested output directory.
7.  Run without requiring manual source-code path edits.

The inference pipeline should use batching and efficient I/O where
appropriate because the official benchmark includes script startup,
model initialization, input reading, inference, and output writing in
the measured end-to-end runtime.

------------------------------------------------------------------------

## 8. Final Test Outputs

After selecting the final model:

``` bash
python evaluate.py \
    --input_dir test \
    --output_dir outputs/denoised
```

The generated files in:

``` text
outputs/denoised/
```

constitute the final denoised test outputs for submission.

The exact filename/extension convention should match the format
specified by the KLA submission portal when the final test package is
submitted.

------------------------------------------------------------------------

## 9. Environment Reproduction

The final environment specification is generated from the environment
used for the submitted model:

``` bash
pip freeze > requirements.txt
```

This file should contain the complete package list used for training and
inference.

------------------------------------------------------------------------

## 10. Evaluation Metrics

The challenge evaluates restoration quality using:

### SSIM

Measures structural similarity between the restored image and ground
truth.

### PSNR / pSNR

Measures pixel-level reconstruction fidelity.

### LPIPS

Measures perceptual similarity using deep image features.

The challenge also evaluates:

### End-to-end inference time

The benchmark uses an NVIDIA H100 GPU and includes:

-   Script startup
-   Model initialization
-   Input I/O
-   Full test-set inference
-   Output I/O

Therefore, inference optimization is considered part of the solution.

------------------------------------------------------------------------

## 11. Reproducibility

Important fixed decisions in the training workflow include:

``` text
Random seed:             42
Validation random_state: 42
Batch size:              16
V3 learning rate:        1e-4
V3 epochs:               25
Augmentation fraction:   0.34
```

The V1 checkpoint:

``` text
checkpoints/best_epoch13.weights.h5
```

is treated as a locked baseline and should not be overwritten.

------------------------------------------------------------------------

## 12. Development vs Final Submission

Development experiments may contain additional notebooks, checkpoints,
experiments, visualizations, and intermediate scripts.

Only the validated final pipeline should be placed in the final
submission package.

The final package should contain:

``` text
Training script
Evaluation script
Final model
Model source code
Test outputs
Environment specification
README
```

------------------------------------------------------------------------

## 13. Official Submission Requirements

The KLA challenge specifies four core technical submission components:

1.  **Standalone evaluation script**
2.  **Training script**
3.  **Denoised test outputs**
4.  **Complete `pip freeze` environment specification**

The evaluation script must accept the test-image directory and output
directory as command-line inputs and run without manual edits.

------------------------------------------------------------------------

## 14. Final Submission Checklist

Before submission, verify:

-   [ ] `train.py` reproduces the submitted training procedure.
-   [ ] `evaluate.py` runs from the command line.
-   [ ] No hard-coded local/Colab paths are required.
-   [ ] `checkpoints/final.weights.h5` is the exact submitted model.
-   [ ] Test outputs were generated using the final checkpoint.
-   [ ] Output files match the required submission format.
-   [ ] `requirements.txt` was generated with `pip freeze`.
-   [ ] Inference works on a clean GPU environment.
-   [ ] Final inference time has been measured.
-   [ ] Validation results for the selected model are recorded.
-   [ ] Final presentation has been exported as PDF according to the
    hackathon template.

------------------------------------------------------------------------

## 15. Project Status

### Completed / Locked

-   ResidualSRCNN architecture
-   V1 baseline checkpoint
-   V3 training strategy
-   Reproducible validation split
-   Real degraded-image training inputs
-   Mild synthetic degradation strategy

### Finalization

The following are finalized only after the V3 validation comparison and
final test run:

-   Final model checkpoint
-   Standalone `evaluate.py`
-   Final denoised test outputs
-   Final inference benchmark
-   `requirements.txt`
-   Final presentation PDF
