# Deep Learning-Based Motion Deblurring and Wagon Number Recognition

An ongoing minor project for detecting freight wagons, locating their printed identification numbers, restoring degraded image regions, and reading the numbers with OCR.

The project is being developed as a staged pipeline so that every part can be tested separately on a laptop with an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

## Project Status

The project is in active development. The detection, crop, restoration baseline, OCR comparison, and manual-label workflow are implemented. Final OCR quality is not yet acceptable.

### Current Next Task

The manual labeling work is complete for the available readable crops. The current files are:

- `Wagon Dataset/number_labels.csv`: 162 verified ground-truth labels used for evaluation.
- `Wagon Dataset/number_labels_aligned.csv`: verified labels sorted and validated by `crop_file`.
- `Wagon Dataset/number_labels_review.csv`: review record containing verified, unreadable, and invalid-crop decisions.

The regeneration step is complete. The next technical task is to improve the number-region boxes and deblurring baseline.

Stage 18 created `number_labels_aligned.csv` using `crop_file` keys, so later evaluation must use this aligned file rather than CSV row positions.

Stage 19 added a CPU-only readiness check. It confirms that aligned labels and verified labels are available, detects that the number-detector weights are still available, and reports that generated OCR comparison files must be recreated. No model training is required for that regeneration step.

### Stage 20: Regenerate evaluation crops and OCR

Existing number-detector weights were reused without retraining. The notebook regenerated 342 number crops and 342 OCR comparison rows using one-image-at-a-time inference. CPU preprocessing and Tesseract OCR were then applied. The deblurred columns remain unavailable until the restoration model is rebuilt.

### Stage 21: Train a crop-focused deblurring baseline

The existing detector was reused to create 342 matching sharp target crops. A small CNN was trained directly on blurred number crops and sharp target crops for 3 epochs with batch size 2. Final crop-training L1 loss was 0.032742. The model is saved locally as `Wagon Dataset/models/crop_deblur_cnn.pt` and is excluded from GitHub.

### Stage 22: Apply crop deblurring and compare OCR

The crop-focused model was applied to all 342 regenerated crops and OCR was rerun. On the 161 matched verified labels:

- Original exact-match accuracy: 0.0.
- OCR-ready exact-match accuracy: 0.0.
- Crop-deblurred exact-match accuracy: 0.0.
- OCR-ready CER: 0.995277.
- Crop-deblurred CER: 0.993506.
- Valid UIC predictions: 0 for all versions.

The crop-focused model gives a small CER improvement but does not yet produce correct full-number OCR results.

### Stage 24: Test low-cost multi-configuration OCR

Two Tesseract page modes were tested on the 162 aligned verified crops only. No GPU or model training was used.

### Stage 25: Evaluate multi-configuration OCR

The selected multi-mode OCR results produced:

- Original: 19 non-empty predictions, CER 0.991202, exact accuracy 0.0.
- OCR-ready: 26 non-empty predictions, CER 0.987683, exact accuracy 0.0.
- Crop-deblurred: 20 non-empty predictions, CER 0.991202, exact accuracy 0.0.

Multi-mode OCR improves the number of non-empty outputs and slightly lowers CER, but it still does not recover complete wagon numbers.

### Final checkpoint

The current project checkpoint is complete through crop-focused restoration and variable-length OCR evaluation. Exact full-number accuracy remains 0.0, while crop-focused deblurring slightly improves CER to 0.993506. The pipeline is functional and measured, but it is not yet a high-accuracy recognition system.

### Stage 23: Evaluate variable-length wagon numbers

The dataset is not fixed at 12 digits. Among the 161 matched verified labels, the ground-truth lengths are:

- 11 digits: 123 labels.
- 10 digits: 21 labels.
- 9 digits: 8 labels.
- 4, 5, 6, and 8 digits: 2 labels each.
- 12 digits: 2 labels.

The new evaluator reports exact match, CER, non-empty predictions, and the number of predictions with 11 or fewer digits. The fixed 12-digit UIC checksum is not used as the primary metric.

Current variable-length results:

- Original non-empty predictions: 10/161.
- OCR-ready non-empty predictions: 9/161.
- Crop-deblurred non-empty predictions: 13/161.
- Exact-match accuracy: 0.0 for all three versions.

### Completed

- Dataset pairing and validation for 252 input/target pairs.
- Wagon and number-region YOLO detector setup, training, and validation.
- Number crop generation, OCR preprocessing, and Tesseract comparison.
- Baseline and crop-focused deblurring CNNs.
- Manual verification and key-aligned ground-truth labels for 162 readable crops.
- Variable-length OCR and multi-configuration OCR evaluation.

### Current measurable data

- Input/target pairs: 252/252.
- Wagon labels: 232 train and 20 validation.
- Number-region labels: 201 train and 25 validation.
- Verified OCR labels: 162.
- Unreadable review rows: 117.
- Invalid-crop review rows: 64.
- GPU: NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

### Wagon detector result

The wagon detector validation run produced the following result:


These values describe the current wagon detector experiment. They do not represent final OCR accuracy or final project accuracy.

### Number-region detector result

The Stage 7 number-region detector was trained separately using the 201 training labels and evaluated on 25 validation labels:


These results are preliminary because the number-region boxes were generated using a rule-based estimate and still require visual correction.

### Important current limitation

The number-region boxes were initially generated as rule-based boxes inside the wagon boxes. They are useful for building the second detector pipeline, but they must be visually checked and improved before reporting final number-detection or OCR results.

The current CNN is a small restoration baseline, not the final deblurring model. A stronger crop-focused restoration model is still required.

## Pipeline Overview

```text
Blurred wagon image
        |
        v
3. Number-region detector
   Finds the printed number area inside the wagon
        |
        v
4. Crop and preprocess the number region
        |
        v
5. Deblur or restore the cropped region
        |
        v
6. OCR
   Converts the number image into text
        |
        v
7. Validation and evaluation
        OCR accuracy, CER, PSNR, and SSIM
```

## Stages

### Stage 1: Install and import the environment

The notebook installs and imports PyTorch, torchvision, Pillow, NumPy, pandas, Matplotlib, scikit-image, OpenCV, pytesseract, and Ultralytics.

### Stage 2: Inspect and validate the dataset

The notebook reads paired files from:

- `Wagon Dataset/input`: degraded or blurred images.
- `Wagon Dataset/output`: corresponding sharper target images.

It checks missing files, image sizes, and creates these manifests:

- `Wagon Dataset/dataset_index.csv`
- `Wagon Dataset/train_pairs.csv`
- `Wagon Dataset/validation_pairs.csv`
- `Wagon Dataset/test_pairs.csv`

### Stage 3: Prepare wagon annotations

The notebook creates the YOLO wagon dataset structure, class file, annotation template, and coordinate preview.

The wagon class is:

```text
0 = wagon
```

### Stage 4: Prepare and validate wagon labels

Wagon bounding boxes are stored in YOLO format. The current wagon labels were cleaned and checked for missing labels, invalid values, and train/validation overlap.

The external `labels/` folder contains the current wagon label files used to create the wagon detector dataset.

### Stage 5: Train and evaluate the wagon detector

A lightweight `yolov8n` model is trained to detect complete wagon boxes. The current notebook configuration is designed for the 4 GB GPU:

- Image size: 640.
- Training batch size: 2.
- Validation batch size: 1.
- DataLoader workers: 2.
- Dataset cache: disabled.
- AMP/mixed precision: enabled.
- CUDA is used when available, otherwise CPU is used.

### Stage 6: Prepare number-region annotations

A separate YOLO dataset is created in `Wagon Dataset/yolo_number`. It has one class:

```text
0 = number_region
```

The number detector is a separate model because the wagon detector finds the complete wagon, while the number detector finds the small printed number area inside it.

### Stage 7: Train the number-region detector

The notebook now contains a separate training and validation cell for the number-region detector. Its outputs are saved under the local `runs/` directory, which is intentionally ignored by Git.

The completed run uses the same low-memory settings as Stage 5:

- Lightweight YOLO model.
- Batch size 1 or 2.
- Workers 2 or fewer.
- Cache disabled.
- AMP enabled.
- Image size selected after checking detection quality and VRAM use.

### Stage 8: Crop number regions

The trained number detector was used to process all 252 input images one at a time. It generated 342 number-region crops in the local `Wagon Dataset/number_crops` folder.

This stage keeps only the small detected regions for later restoration and OCR, which reduces memory use on the 4 GB GPU. The crop count can be greater than the image count because an image may contain more than one detected number region.

### Stage 9: Prepare OCR-ready crop variants

Each of the 342 number-region crops was converted to grayscale, enlarged by 3x, lightly smoothed, and thresholded with Otsu's method. The results are stored locally in `Wagon Dataset/ocr_ready`.

This is a baseline preprocessing step, not the final deblurring model. OCR accuracy is not reported yet because real ground-truth wagon numbers have not been added.

### Stage 10: Run baseline OCR

The notebook processed 342 OCR-ready crops in the experiment run and wrote predictions to `Wagon Dataset/ocr_baseline.csv`. Generated OCR outputs were later cleaned from the workspace and can be regenerated.

### Stage 11: Train and evaluate a lightweight deblurring model

A small three-convolution CNN was trained from scratch on the paired blurred/sharp images. To fit the RTX 3050 with 4 GB VRAM, images were resized to 256x256, training used batch size 2, one data-loading worker, and 5 epochs.

Current baseline result on 25 validation images:

- Final training L1 loss: 0.039804.
- Validation PSNR: 25.907.
- Validation SSIM: 0.8519.
- Model file: local `Wagon Dataset/models/small_deblur_cnn.pt`.

These are baseline restoration results. The model has since been applied to the detected number crops and compared against the original degraded crops.

### Stage 12: Apply deblurring to number-region crops

The Stage 11 baseline model was applied to all 342 detected number-region crops one at a time. The restored crops were saved locally in `Wagon Dataset/deblur_outputs` with their original crop dimensions preserved.

This output is ready for visual comparison and later OCR testing. It is not yet the final deblurring model result.

### Stage 13: Compare crop versions

All 342 original, OCR-ready, and deblurred crops were compared. The report uses Laplacian sharpness and mean pixel change, plus a six-sample visual contact sheet.

Current average sharpness values:

- Original crops: 57.6086.
- OCR-ready crops: 2490.6267.
- Deblurred crops: 57.5158.

The OCR-ready value is expected to be much higher because thresholding creates strong edges. These values are diagnostic only and do not prove that the deblurred model is better. Final quality requires aligned sharp number-region targets and OCR evaluation.

### Stage 14: Compare OCR inputs

Tesseract was available during this run, so all 342 original, OCR-ready, and deblurred crops were processed. The output is stored in `Wagon Dataset/ocr_input_comparison.csv`.

Non-empty OCR predictions were returned for:

- Original crops: 20.
- OCR-ready crops: 22.
- Deblurred crops: 16.

These counts are not OCR accuracy because ground-truth wagon numbers have not been added. They show that the current baseline deblurring model did not improve the number of non-empty OCR outputs over the simple OCR-ready preprocessing.

### Stage 15: Prepare ground-truth OCR evaluation

The verified file now exists and contains 162 rows. The current OCR comparison matched 161 of them; `image_88_region_01.png` has a verified label but is absent from the saved OCR comparison and is excluded until it is regenerated.

Current evaluation on the 161 matched rows:

- Original exact-match accuracy: 0.0.
- OCR-ready exact-match accuracy: 0.0.
- Deblurred exact-match accuracy: 0.0.
- Original, OCR-ready, and deblurred valid UIC predictions: 0.

These results show that the current OCR baseline is not yet accurate enough for final recognition. The deblurred score is not a valid model result in this run because the deblurring model was intentionally not regenerated.

### Stage 16: Review OCR candidates

The notebook created `Wagon Dataset/number_labels_review.csv` during the review workflow. It contains OCR candidates plus the manual `verified_ground_truth` and `review_status` decisions.

The real numbers were read from the sharp reference crops and entered by a person. OCR candidates must not be copied blindly as ground truth.

### Stage 17: Create sharp reference crops for labeling

The blurred number crops were not readable enough for reliable annotation. The notebook can transfer the same detector coordinates to the paired sharp output images and generate clear reference crops in `Wagon Dataset/number_target_crops`. Those generated crops were cleaned after labeling and are not currently present.

Use the matching filename from this sharp-reference folder when filling `verified_ground_truth`. These reference crops are generated artifacts and are kept outside GitHub.

### Optional future improvements

- Improve and manually verify the number-region boxes, then retrain Stage 7 with corrected labels.
- Train the crop-focused deblurring model for more epochs while respecting the 4 GB VRAM limit.
- Test stronger OCR preprocessing or a dedicated OCR model.
- Add more readable labeled samples if available.
- Test failure cases such as missing detections and false detections.

## Repository Structure

```text
Minor Project/
|
|-- wagon_deblurring_pipeline.ipynb   # Main staged notebook
|-- requirements.txt                  # Python dependencies
|-- README.md                         # Project documentation
|-- .gitignore                        # GitHub upload rules
|-- labels/                           # Current wagon YOLO labels
|-- Wagon Dataset/
|   |-- dataset_index.csv             # Pair metadata and split information
|   |-- train_pairs.csv               # Training manifest
|   |-- validation_pairs.csv          # Validation manifest
|   |-- test_pairs.csv                # Test manifest
|   |-- wagon_annotations_template.csv
|   |-- number_labels.csv           # 162 verified OCR ground-truth rows
|   |-- number_labels_aligned.csv   # Key-aligned verified labels
|   |-- number_labels_review.csv    # Manual review decisions and OCR candidates
|   |-- yolo_wagon/                    # Wagon detector config and labels
|   |-- yolo_number/                   # Number detector config and labels
|   |-- input/                         # Local degraded images, ignored by Git
|   |-- output/                        # Local target images, ignored by Git
|   |-- restored/                      # Future restoration outputs, ignored by Git
|   |-- models/                        # Future model files, ignored by Git
|-- runs/                              # Local YOLO results, ignored by Git
|-- weights/                           # Local downloaded weights, ignored by Git
```

Large image files, trained weights, prediction crops, and experiment logs are ignored so the GitHub repository remains manageable. The dataset should be shared separately or downloaded from the agreed project source.

The local workspace was cleaned after the current experiments. Generated crops, duplicate YOLO image copies, model outputs, previews, and experiment reports were removed; they can be regenerated by rerunning the relevant notebook stages. Source images, sharp targets, labels, manifests, and verified OCR labels were preserved.

## Setup

Use Python 3.10 or newer if possible.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU use, confirm PyTorch can see CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected hardware for the current configuration:

```text
NVIDIA RTX 3050 Laptop GPU
4 GB VRAM
CUDA-enabled PyTorch
```

If CUDA is unavailable, the notebook falls back to CPU, but YOLO training will be much slower.

## Running the Notebook

Open `wagon_deblurring_pipeline.ipynb` in VS Code with the Python and Jupyter extensions installed.

Run the notebook cells in order. Do not skip dataset validation or label validation. Before training, confirm that the expected images and label files are present locally.

The notebook uses a fallback Windows dataset path, but the preferred layout is to open the notebook from the project folder and keep `Wagon Dataset` beside it.

## OCR Notes

`pytesseract` is only the Python wrapper. The Tesseract OCR application must also be installed separately on Windows and available at one of the paths checked by the notebook.

OCR accuracy must not be reported until real ground-truth number labels are supplied. The checksum function can reject an invalid candidate, but it cannot recover a missing or unreadable digit.

## Low-VRAM Design Rules

All future stages must respect the 4 GB VRAM limit:

- Prefer small models and cropped inputs.
- Use batch size 1 or 2.
- Use no more than 2 DataLoader workers by default.
- Keep caching disabled.
- Use AMP/mixed precision where supported.
- Release unused models and tensors before starting another model.
- Avoid loading the complete high-resolution dataset into GPU memory.
- Track memory use and runtime along with accuracy.
- Increase model size only if a measured experiment justifies it.

## Reproducibility

The project uses fixed random seeds where applicable and stores dataset split metadata in CSV files. Generated model weights and experiment outputs are not committed by default because they are large and machine-specific.

When sharing an experiment, record:

- Model name and version.
- Dataset split used.
- Image size.
- Batch size.
- Number of epochs.
- GPU or CPU used.
- Validation metrics.
- Output run directory.

## Team Workflow

Before changing the notebook:

1. Read the current README and notebook stage headings.
2. Keep one implementation per stage.
3. Do not commit generated images, model weights, or run logs.
4. Record new metrics and limitations in this README.
5. Keep experimental code clearly separated from final pipeline code.

## License and Dataset

No license or dataset redistribution terms have been selected yet. Add the appropriate license and dataset attribution before publishing the repository publicly.
