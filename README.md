# Deep Learning-Based Motion Deblurring and Wagon Number Recognition

An ongoing minor project for detecting freight wagons, locating their printed identification numbers, restoring degraded image regions, and reading the numbers with OCR.

The project is being developed as a staged pipeline so that every part can be tested separately on a laptop with an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

## Project Status

The project is in active development.

### Completed

- Dataset inspection and input/output pairing.
- Reproducible train, validation, and test metadata generation.
- Image-size and missing-pair checks.
- YOLO wagon annotation folder setup.
- Wagon label integration and validation.
- Wagon detector training and validation using a lightweight YOLOv8n model.
- Number-region dataset folder setup.
- Initial number-region boxes generated from the validated wagon boxes.
- Number-region YOLO experiment trained separately from the wagon detector.
- GPU and low-VRAM compatibility checks.
- GitHub documentation and ignore rules.

### Current measurable data

- Paired input images: 252.
- Paired target images: 252.
- Wagon training labels: 232.
- Wagon validation labels: 20.
- Number-region training labels: 201.
- Number-region validation labels: 25.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU.
- Available VRAM: 4 GB.
- CUDA: available through PyTorch.

### Wagon detector result

The wagon detector validation run produced the following result:

- Precision: 0.997.
- Recall: 1.000.
- mAP50: 0.995.
- mAP50-95: 0.893.

These values describe the current wagon detector experiment. They do not represent final OCR accuracy or final project accuracy.

### Important current limitation

The number-region boxes were initially generated as rule-based boxes inside the wagon boxes. They are useful for building the second detector pipeline, but they must be visually checked and improved before reporting final number-detection or OCR results.

The CNN deblurring prototype was intentionally removed from the notebook. The restoration model will be designed and implemented in a later stage from scratch.

## Pipeline Overview

```text
Blurred wagon image
        |
        v
1. Dataset validation and pairing
        |
        v
2. Wagon detector
   Finds the complete wagon area
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
   UIC checksum, OCR accuracy, CER, PSNR, and SSIM
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

An initial number-region detector experiment was run using the generated number-region labels. Its outputs were saved under the local `runs/` directory, which is intentionally ignored by Git.

The training code should be organized into the notebook in the next cleanup step. It must use the same low-memory settings as Stage 5:

- Lightweight YOLO model.
- Batch size 1 or 2.
- Workers 2 or fewer.
- Cache disabled.
- AMP enabled.
- Image size selected after checking detection quality and VRAM use.

### Later stages

The following work remains:

- Visually inspect and correct number-region boxes.
- Finalize and document the Stage 7 training cell.
- Generate reliable number-region crops.
- Implement the deblurring/restoration model from scratch.
- Compare the restored crop with the original degraded crop and sharp target.
- Install and configure Tesseract OCR where required.
- Run OCR on original, restored, and target/crop images.
- Add real ground-truth number labels in `number_labels.csv`.
- Measure OCR accuracy, character error rate, and valid UIC-number rate.
- Test failure cases such as missing detections, unreadable numbers, and false detections.
- Prepare final plots, tables, conclusions, and presentation material.

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
