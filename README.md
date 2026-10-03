# Deep Learning-Based Motion Deblurring and Wagon Number Recognition

## 1. Project summary

This project is a staged computer-vision pipeline for processing degraded freight-wagon images. Its intended end-to-end behavior is:

1. Pair a blurred/degraded wagon image with its sharper target image.
2. Detect the complete wagon.
3. Detect the printed identification-number region inside the wagon.
4. Crop and preprocess the number region.
5. Restore or deblur the cropped region.
6. Read the wagon number with OCR.
7. Compare OCR predictions with verified ground truth.

The main implementation is the Jupyter notebook [wagon_deblurring_pipeline.ipynb](./wagon_deblurring_pipeline.ipynb). The project is an experimental research pipeline, not a finished production recognition system. The current pipeline executes and produces measurable intermediate outputs, but its final number-recognition accuracy is not acceptable.

## 2. Current status

### Overall status

| Area | Status |
|---|---|
| Input/target pairing | Completed for 252 pairs |
| Dataset manifests | Completed |
| Wagon YOLO dataset preparation | Completed |
| Wagon detector training workflow | Implemented |
| Number-region YOLO dataset preparation | Completed |
| Number-region detector training workflow | Implemented |
| Number-region crop generation | Completed for the latest evaluation run |
| OCR preprocessing | Implemented |
| Baseline OCR comparison | Completed |
| Full-image deblurring baseline | Completed |
| Crop-focused deblurring baseline | Completed |
| Manual ground-truth review | Completed for available readable crops |
| Variable-length OCR evaluation | Completed |
| Multi-configuration OCR evaluation | Completed |
| Final OCR quality | **Not acceptable** |
| Production/export readiness | **Not ready** |

### Current checkpoint

The latest documented checkpoint is complete through crop-focused restoration, variable-length evaluation, and low-cost multi-configuration OCR. The most important result is:

- Exact full-number accuracy is `0.0` for all tested OCR versions.
- Crop-focused deblurring lowers CER only marginally.
- The number-region boxes still need visual correction.
- The current CNN is a baseline and must not be presented as a final deblurring model.

## 3. Hardware and runtime assumptions

The notebook was designed for a laptop with:

- NVIDIA RTX 3050 Laptop GPU.
- 4 GB VRAM.
- Windows.
- CPU fallback when CUDA is unavailable.

The training configuration intentionally uses small batches, disabled dataset caching, low worker counts, and mixed precision where supported. A larger GPU can use larger images, batches, and stronger models, but those settings have not been validated in this repository.

## 4. Technology stack

The project uses:

- Python.
- Jupyter Notebook.
- PyTorch and torchvision.
- Ultralytics YOLO.
- OpenCV.
- Pillow.
- NumPy and pandas.
- scikit-image for PSNR and SSIM.
- Matplotlib for previews and comparisons.
- pytesseract with a local Tesseract installation for OCR.

The dependency list is maintained in [requirements.txt](./requirements.txt).

## 5. Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open the notebook:

```powershell
jupyter notebook wagon_deblurring_pipeline.ipynb
```

The notebook also contains a setup installation cell for interactive use. Installing from `requirements.txt` first is preferred for reproducibility.

### Tesseract requirement

`pytesseract` is only a Python wrapper. The Tesseract executable must also be installed locally. The notebook checks:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
```

It also checks whether `tesseract` is available on `PATH`. If Tesseract is missing, OCR rows are written with an unavailable status rather than silently being treated as successful predictions.

## 6. Input data contract

The notebook expects paired images in:

```text
Wagon Dataset/input/
Wagon Dataset/output/
```

The input and target image must have the same filename. The input is the degraded/blurred image and the output is the sharper reference image.

The pairing stage:

- Lists all PNG files.
- Detects missing targets.
- Detects missing inputs.
- Records image dimensions.
- Records whether each pair has matching dimensions.
- Randomly assigns deterministic train, validation, and test splits with seed `42`.
- Writes the split metadata to CSV manifests.

The current dataset checkpoint contains:

- 252 valid input/target pairs.
- 232 wagon training labels.
- 20 wagon validation labels.
- 201 number-region training labels.
- 25 number-region validation labels.
- 162 verified OCR review rows.
- 117 rows marked unreadable.
- 64 rows marked invalid crop.

The raw input and target images are intentionally ignored by Git because of repository size. They must be supplied locally before rerunning the pipeline.

## 7. Pipeline architecture

```text
Degraded wagon image
        |
        v
Dataset pairing and split validation
        |
        v
Wagon detector (YOLO)
        |
        v
Number-region detector (YOLO)
        |
        v
Number-region crop
        |
        +--------------------+
        |                    |
        v                    v
OCR preprocessing       Crop deblurring CNN
        |                    |
        +---------+----------+
                  |
                  v
              Tesseract OCR
                  |
                  v
       Ground-truth evaluation:
       exact match, CER, non-empty output,
       variable-length analysis, UIC check
```

The wagon detector and number-region detector are separate models. The first model identifies a whole wagon; the second identifies the small printed number area. A detector failure directly affects every later stage.

## 8. Notebook stage documentation

### Stage 1: Install and import dependencies

The notebook installs or imports PyTorch, torchvision, Pillow, NumPy, pandas, Matplotlib, scikit-image, OpenCV, pytesseract, and Ultralytics. Random seeds are set to `42` for Python and PyTorch.

The notebook selects CUDA when available and otherwise uses CPU.

### Stage 2: Validate image pairs and create manifests

The notebook reads the input and target folders, validates matching filenames and dimensions, shuffles the metadata deterministically, and creates:

- `Wagon Dataset/dataset_index.csv`
- `Wagon Dataset/train_pairs.csv`
- `Wagon Dataset/validation_pairs.csv`
- `Wagon Dataset/test_pairs.csv`

### Stage 3: Prepare wagon annotations

The notebook creates the wagon YOLO directory structure, writes the class file, writes a CSV annotation template, and creates a coordinate preview.

The wagon detector has one class:

```text
0 = wagon
```

### Stage 4: Copy wagon images and save labels

Images are copied into the YOLO train, validation, and test folders. `pixel_box_to_yolo()` converts pixel coordinates to normalized YOLO coordinates. `save_wagon_labels()` writes wagon labels, and `check_yolo_labels()` validates class IDs, field counts, and normalized values.

### Stage 5: Train and validate the wagon detector

The notebook trains a lightweight `yolov8n` model with low-memory settings:

- Image size: `640`.
- Batch size: `2`.
- Workers: `2`.
- Cache: disabled.
- AMP: enabled.
- CUDA if available, otherwise CPU.

The training output is written to a local `runs/` directory, which is ignored by Git. The notebook includes validation logic, but the repository documentation does not contain a complete reproducible detector metric table. Those metrics must be regenerated and recorded before the detector can be reported as a final result.

### Stage 6: Prepare number-region annotations

The notebook creates a second YOLO dataset under `Wagon Dataset/yolo_number`.

The number detector has one class:

```text
0 = number_region
```

The number-region images are copied from the paired input images. The number labels are separate from wagon labels because number-region detection is a different task.

### Stage 7: Train and validate the number-region detector

The notebook trains a separate lightweight YOLO model using the same general low-memory strategy. It can load a saved detector from candidate local paths and validate it on the validation split.

The latest number-region labels were initially based on rule-generated boxes and require visual correction. Therefore, current number-detector results must be treated as preliminary.

### Stage 8: Generate number crops

The number detector is applied one image at a time. Each detected region is clipped to image boundaries and saved using a stable name such as:

```text
image_1_region_01.png
```

The latest regeneration produced 342 crops from 252 input images. The count is larger than the image count because an image can contain multiple detections.

### Stage 9: Prepare OCR-ready crops

Each crop is:

1. Converted to grayscale.
2. Enlarged by 3x.
3. Smoothed with a Gaussian blur.
4. Thresholded using Otsu's method.

The results are saved in `Wagon Dataset/ocr_ready/`. This is baseline preprocessing, not a learned restoration method.

### Stage 10: Run baseline OCR

Tesseract is run with a numeric whitelist and page segmentation mode 7:

```text
--psm 7 -c tessedit_char_whitelist=0123456789
```

The notebook writes OCR predictions and status fields. Missing Tesseract, unreadable images, and successful rows are represented explicitly.

### Stage 11: Train the full-image deblurring baseline

`SmallDeblurCNN` is a small three-convolution CNN:

- 3 input channels.
- 16 feature channels.
- ReLU activations.
- 3 output channels.
- Sigmoid output.

Training uses paired full images resized to `256 x 256`, batch size `2`, one data-loading worker, Adam with learning rate `0.001`, L1 loss, and 5 epochs.

The recorded baseline result on 25 validation images was:

- Final training L1 loss: `0.039804`.
- Validation PSNR: `25.907`.
- Validation SSIM: `0.8519`.

The local model path is:

```text
Wagon Dataset/models/small_deblur_cnn.pt
```

### Stage 12: Apply full-image deblurring to crops

The full-image CNN is applied to detected number crops. Each crop is resized to the model input size and resized back to its original crop dimensions after inference. The output is written to `Wagon Dataset/deblur_outputs/`.

This method is only a baseline because the model was trained on full images, not specifically on number-region crops.

### Stage 13: Compare crop versions

The notebook compares original, OCR-ready, and deblurred crops using:

- Laplacian variance as a sharpness diagnostic.
- Mean absolute pixel change.
- A six-sample visual contact sheet.

Recorded mean sharpness values:

| Version | Mean sharpness |
|---|---:|
| Original | 57.6086 |
| OCR-ready | 2490.6267 |
| Deblurred | 57.5158 |

The OCR-ready value is expected to be high because thresholding creates strong edges. Sharpness alone does not demonstrate better recognition quality.

### Stage 14: Compare OCR inputs

The notebook runs OCR on original, OCR-ready, and deblurred crops. In the documented baseline run, non-empty predictions were:

| Version | Non-empty predictions |
|---|---:|
| Original | 20 |
| OCR-ready | 22 |
| Deblurred | 16 |

These are output counts, not accuracy measurements.

### Stages 15-18: Create and verify OCR ground truth

The manual review workflow:

1. Creates a ground-truth template.
2. Creates a review CSV containing OCR candidates.
3. Generates sharp reference crops using the detector coordinates mapped onto the paired sharp images.
4. Allows a human to enter the real printed number.
5. Records `verified`, `unreadable`, or `invalid_crop`.
6. Aligns verified labels by `crop_file`, never by CSV row position.

The important files are:

- `Wagon Dataset/number_labels_review.csv`
- `Wagon Dataset/number_labels.csv`
- `Wagon Dataset/number_labels_aligned.csv`

The aligned file is the authoritative evaluation input for later stages.

### Stage 19: Check final evaluation inputs

The notebook checks for aligned labels, verified labels, detector weights, and OCR comparison files. Missing generated artifacts are reported so they can be regenerated without retraining every model.

### Stage 20: Regenerate evaluation crops and OCR

Existing number-detector weights are reused. The notebook regenerates number crops and OCR-ready images one image at a time and writes the comparison file. This stage is intended to repair stale or missing generated outputs without retraining YOLO.

### Stage 21: Train crop-focused deblurring

The crop-focused baseline creates sharp target crops using the paired target images and the detector coordinates. It trains `CropDeblurCNN` directly on blurred number crops and sharp target crops.

Recorded configuration:

- 342 blurred crops.
- 342 sharp target crops.
- 3 epochs.
- Batch size `2`.
- L1 loss.
- Final training L1 loss: `0.032742`.

The local model path is:

```text
Wagon Dataset/models/crop_deblur_cnn.pt
```

The weights are ignored by Git and must be regenerated locally.

### Stage 22: Apply crop deblurring and compare OCR

The crop-focused model is applied to all regenerated crops and OCR is rerun. On 161 matched verified labels:

| Version | Exact-match accuracy | Character error rate | Valid UIC predictions |
|---|---:|---:|---:|
| Original | 0.0 | Not retained in this checkpoint | 0 |
| OCR-ready | 0.0 | 0.995277 | 0 |
| Crop-deblurred | 0.0 | 0.993506 | 0 |

The small CER improvement is not sufficient to claim successful recognition.

### Stage 23: Evaluate variable-length wagon numbers

The dataset is not fixed at 12 digits. Among the 161 matched labels:

- 11 digits: 123 labels.
- 10 digits: 21 labels.
- 9 digits: 8 labels.
- 4, 5, 6, and 8 digits: 2 labels each.
- 12 digits: 2 labels.

The evaluator reports exact match, character error rate, non-empty predictions, and predictions with 11 or fewer digits. Fixed 12-digit UIC validation is not used as the primary metric because the dataset contains variable-length values.

Non-empty predictions:

| Version | Non-empty predictions |
|---|---:|
| Original | 10/161 |
| OCR-ready | 9/161 |
| Crop-deblurred | 13/161 |

Exact match remained `0.0` for all versions.

### Stages 24-25: Multi-configuration OCR

The notebook tests Tesseract page modes 6 and 7 on the verified crops and selects the longest candidate for each version.

Recorded results:

| Version | Non-empty predictions | CER | Exact-match accuracy |
|---|---:|---:|---:|
| Original | 19 | 0.991202 | 0.0 |
| OCR-ready | 26 | 0.987683 | 0.0 |
| Crop-deblurred | 20 | 0.991202 | 0.0 |

Multi-mode OCR produces more non-empty strings and a slight CER improvement, but it still does not recover complete wagon numbers.

## 9. Evaluation definitions

### Exact-match accuracy

A prediction is correct only when the normalized predicted digit string exactly equals the normalized verified ground-truth string.

### Character error rate

The notebook computes:

```text
total Levenshtein distance / total ground-truth character count
```

Predictions are normalized to digits before comparison.

### Non-empty prediction count

Counts predictions that contain at least one digit. This measures whether OCR returns anything, not whether the result is correct.

### UIC validation

The notebook contains a 12-digit UIC checksum helper for diagnostic use. It is not the primary metric because the verified dataset contains variable-length numbers.

## 10. Repository structure

```text
Minor Project/
|
|-- wagon_deblurring_pipeline.ipynb
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- labels/
|   `-- wagon YOLO label files
|-- Wagon Dataset/
|   |-- input/                    # local degraded images; ignored by Git
|   |-- output/                   # local sharp targets; ignored by Git
|   |-- dataset_index.csv
|   |-- train_pairs.csv
|   |-- validation_pairs.csv
|   |-- test_pairs.csv
|   |-- wagon_annotations_template.csv
|   |-- number_labels.csv
|   |-- number_labels_aligned.csv
|   |-- number_labels_review.csv
|   |-- yolo_wagon/
|   |   |-- data.yaml
|   |   |-- classes.txt
|   |   `-- labels/
|   |-- yolo_number/
|   |   |-- data.yaml
|   |   |-- classes.txt
|   |   `-- labels/
|   `-- models/                   # local model weights; ignored by Git
|-- runs/                         # local YOLO outputs; ignored by Git
`-- weights/                      # downloaded model weights; ignored by Git
```

Generated folders such as crops, OCR-ready images, deblurred images, model weights, YOLO runs, and comparison CSVs are intentionally ignored or locally regenerated. Do not assume a fresh clone contains them.

## 11. Reproducible execution order

Run the notebook cells in order for a full experiment:

1. Install dependencies and import libraries.
2. Validate input/target pairs.
3. Create manifests and inspect sample image quality.
4. Prepare wagon YOLO files and labels.
5. Train and validate the wagon detector.
6. Prepare number-region YOLO files and labels.
7. Train and validate the number detector.
8. Generate number crops.
9. Generate OCR-ready crops.
10. Run baseline OCR.
11. Train and evaluate the full-image deblurring baseline.
12. Apply full-image deblurring.
13. Compare crop versions and OCR inputs.
14. Create and complete manual ground-truth review.
15. Align verified labels.
16. Regenerate evaluation outputs if stale or missing.
17. Train the crop-focused deblurring baseline.
18. Apply crop deblurring.
19. Run variable-length and multi-configuration OCR evaluation.

For evaluation-only work, use the later readiness/regeneration stages after confirming that detector weights, source images, and verified labels are present.

## 12. Completed work

- Established a paired blurred/sharp image workflow.
- Added deterministic dataset splitting and CSV manifests.
- Added image-size and missing-file validation.
- Added YOLO wagon annotation preparation.
- Added YOLO number-region annotation preparation.
- Added label conversion and validation helpers.
- Implemented wagon detector training and validation.
- Implemented number-region detector training and validation.
- Implemented one-image-at-a-time crop generation.
- Implemented grayscale, enlargement, blur, and Otsu OCR preprocessing.
- Implemented Tesseract OCR with explicit availability/error statuses.
- Implemented a full-image CNN deblurring baseline.
- Implemented a crop-focused CNN deblurring baseline.
- Added PSNR and SSIM restoration metrics.
- Added crop sharpness and pixel-change diagnostics.
- Added human review and verified OCR labels.
- Added key-aligned ground-truth evaluation.
- Added variable-length number evaluation.
- Added multi-configuration OCR evaluation.
- Removed duplicate notebook setup and helper definitions for export.

## 13. Known limitations

### 13.1 Number-region labels are not fully trustworthy

The initial number-region boxes were rule-based estimates inside wagon boxes. Some rows are marked invalid or unreadable. The number detector therefore has label noise and may crop the wrong area.

### 13.2 The detector metric record is incomplete

The notebook can train and validate both YOLO models, but complete detector precision, recall, mAP, and per-split results are not preserved in the project documentation. A final report must regenerate these metrics and record the exact model, dataset version, and split.

### 13.3 OCR quality is currently unacceptable

Exact-match accuracy is zero across all tested versions. Most crops produce no useful OCR result, and the low CER improvements are not enough for dependable identification.

### 13.4 The deblurring model is too small

The CNN is a three-convolution baseline trained for very few epochs. It does not model realistic motion blur, text structure, illumination variation, or detector-coordinate uncertainty.

### 13.5 Full-image training is not aligned with the OCR task

The first deblurring model is trained on full images but evaluated on small number crops. It can improve general image appearance without restoring the characters needed by OCR.

### 13.6 Crop correspondence can be wrong

Sharp target crops are generated by transferring detector coordinates from the degraded image to the target image. If input and target dimensions differ, or if the detector box is inaccurate, the blurred and sharp crops may not correspond precisely.

### 13.7 The test split is not a complete final benchmark

The pipeline creates a test manifest, but the documented OCR evaluation is based on matched verified labels, not a complete independently held-out recognition benchmark. Model selection and final reporting need a strict train/validation/test protocol.

### 13.8 OCR ground truth is limited

Only 162 rows were manually reviewed, and 161 matched the comparison file in the documented evaluation. This is useful for debugging but too small and incomplete for a production accuracy claim.

### 13.9 Generated artifacts are not portable

Input images, target images, model weights, YOLO runs, crops, OCR outputs, and previews are ignored or generated locally. Another machine must recreate them using the same source data and compatible model versions.

### 13.10 Tesseract is not specialized for this task

Tesseract with two page modes is a low-cost baseline. It is sensitive to crop quality, text orientation, font, contrast, blur, and character spacing. It is not a substitute for a trained text detector/recognizer.

## 14. Exactly what must be fixed

The following items are required before calling the project complete or export-ready.

### Priority 1: Correct the number-region dataset

1. Review every number-region bounding box against the sharp target image.
2. Fix boxes that include background, cut off characters, or point to the wrong region.
3. Remove duplicate or unusable crops.
4. Record annotation provenance and a dataset version.
5. Recreate train, validation, and test label counts after correction.

### Priority 2: Retrain and document both detectors

1. Retrain the number detector using corrected boxes.
2. Validate on a genuinely held-out split.
3. Record precision, recall, mAP50, mAP50-95, false positives, missed detections, and representative visual examples.
4. Confirm that the wagon detector does not leak images between splits.
5. Save the exact model configuration and package versions used for the reported run.

### Priority 3: Build a crop-focused restoration model

1. Train only on correctly aligned blurred/sharp number crops.
2. Increase training duration with early stopping or a validation-based checkpoint.
3. Compare a stronger architecture, such as a residual CNN or U-Net-style model, against the current baseline.
4. Add realistic blur, noise, brightness, contrast, and perspective augmentation.
5. Measure crop PSNR and SSIM on held-out target crops.
6. Evaluate character preservation, not only visual sharpness.

### Priority 4: Improve text recognition

1. Compare several preprocessing variants instead of selecting only the longest OCR string.
2. Test deskewing, perspective correction, adaptive thresholding, morphology, denoising, and contrast normalization.
3. Test a text-specific OCR model or a trained character recognizer.
4. Keep all candidate predictions and confidence values.
5. Add rejection logic for low-confidence or structurally invalid predictions.
6. Do not use checksum rules to overwrite OCR predictions.

### Priority 5: Expand and isolate evaluation data

1. Increase the verified ground-truth set beyond 162 rows.
2. Resolve the unmatched `image_88_region_01.png` row by regenerating the comparison output.
3. Keep train, validation, and test labels strictly separated.
4. Report metrics separately for readable, unreadable, invalid-crop, and missed-detection cases.
5. Report exact match, CER, normalized edit distance, character accuracy, empty-output rate, and detection recall.

### Priority 6: Make the pipeline reproducible

1. Pin dependency versions in `requirements.txt`.
2. Add a configuration section for dataset paths, model paths, confidence thresholds, image size, seed, and device.
3. Avoid relying on notebook state or previously executed cells.
4. Add a clean evaluation entry point that verifies all required inputs before running.
5. Record model hashes, dataset version, and experiment settings with each result.
6. Verify the complete notebook from a clean kernel on a second machine.

### Priority 7: Prepare a defensible export

Before final submission or deployment, the project should include:

- Corrected and versioned number-region annotations.
- Reproducible detector metrics.
- A held-out OCR benchmark.
- A stronger crop-focused restoration comparison.
- A recognized failure-case gallery.
- A clear statement of expected accuracy and unsupported inputs.
- Reproducible environment instructions.
- No claims of high-accuracy recognition while exact-match accuracy remains zero.

## 15. Recommended next experiment

The next experiment should not be another OCR configuration sweep. The highest-value sequence is:

1. Correct a representative batch of number-region boxes.
2. Retrain the number detector.
3. Visually inspect regenerated crops.
4. Train the crop-focused restoration baseline on corrected pairs.
5. Evaluate a small set of preprocessing variants.
6. Measure results on a held-out verified subset.

This isolates whether the main failure is caused by incorrect localization, insufficient restoration, or OCR limitations. Without that isolation, additional model changes will not produce a reliable conclusion.

## 16. Final project assessment

The repository demonstrates a complete experimental workflow from paired image preparation through detection, restoration, OCR, manual verification, and evaluation. It is suitable as a minor-project research prototype and a foundation for further experiments.

It is not yet suitable for:

- Production wagon-number recognition.
- Reporting a reliable final OCR accuracy.
- Claiming successful deblurring-based recognition.
- Comparing models without first correcting the number-region labels.
- Exporting model outputs as authoritative wagon identities.

The project should be described honestly as a functional pipeline with a failed current recognition baseline and a clearly identified remediation path.
