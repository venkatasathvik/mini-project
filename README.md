# OpenPulse: Edge-AI FPGA Pipeline for Satellite Telemetry

## Project Overview
The OpenPulse project demonstrates a highly constrained, hardware-aware AI pipeline designed for bare-metal FPGA deployment. The objective is to perform binary image classification (Cloud vs. Clear terrain) directly on satellite telemetry using a custom **F=4 Micro-CNN** mapped to an INT8 systolic array in Verilog.

Because standard floating-point Neural Networks cannot be deployed directly to small FPGAs without memory overflows, this project features a strict Python software pipeline that handles 8-bit integer quantization and specific memory-mapping exports (`.coe` files) for Xilinx Vivado Block RAM (BRAM).

---

##  Data Sources & Preprocessing
To train and test the model within hardware limits, the image data must be single-channel grayscale, exactly 128x128 pixels, and scaled strictly to positive 8-bit integers (0 to 127).

1. **Training Data (Kaggle 38-Cloud Dataset):**
   - **Source:** Landsat 8 satellite captures.
   - **Usage:** Used as the primary training baseline. Our preprocessing script extracts exactly 1,000 clear and 1,000 cloudy tiles to create a perfectly balanced 2,000-image dataset.
   - **Why 2,000?** A tiny 4-filter CNN lacks the mathematical capacity to absorb tens of thousands of images (underfitting) but will memorize too few (overfitting). 2,000 is the mathematical sweet spot.

2. **Testing Data (SkyPulse Telemetry):**
   - **Source:** Live local satellite telemetry captured via rooftop ground station.
   - **Usage:** Because only sample captures are currently available, this dataset is reserved strictly for **unseen data validation**. It proves the CNN can generalize from academic Kaggle data to real-world hardware captures.

---

## Hardware-Aware Architecture (F=4 Micro-CNN)
The neural network is deliberately stunted to fit within the FPGA's BRAM limitations and to avoid unsupported hardware operations (like floating-point exponential math).

* **Input:** 128x128 Grayscale Image (INT8: 0 to 127)
* **Layer 1 (Conv2D):** 4 Filters ($F=4$), 3x3 Kernel, No Bias.
* **Activation:** ReLU (Hardware equivalent: zero out negative integers).
* **Layer 2 (MaxPool):** 2x2 pooling with stride 2. Output becomes 64x64.
* **Layer 3 (Flatten):** 4 channels * 64 * 64 = 16,384 nodes.
* **Layer 4 (Dense/FC):** Fully Connected down to **1 output node** (Binary Classification). 
* **Quantization:** Symmetric Post-Training Quantization (PTQ). Weights are clamped strictly between -128 and +127 with a Zero-Point of 0.

---

##  Directory Structure & Important Files

Based on the project workspace, here are the critical files and what they do:

```text
mini-project/
├── dataset/                    # The master folder for the balanced 2,000 training images.
│   ├── clear/                  
│   └── cloudy/                 
├── project/
│   ├── output files/           # Target destination for Vivado hardware files.
│   │   ├── conv1_weights.coe   # Hex file containing the 36 Conv2D weights.
│   │   └── fc_weights.coe      # Hex file containing the 16,384 Dense layer weights.
│   └── trainer/
│       ├── slicing.py          # Script: Slices, thresholds, and scales the 38-Cloud dataset into 128x128 tiles.
│       └── trainer.py          # Script: The core AI engine. Trains the model, tests accuracy, and exports .coe files.
├── sample dataset/             # Raw downloads folder (Kaggle 38-Cloud folders & raw SkyPulse captures).
└── test_data/                  # Folder for SkyPulse accuracy testing.
    ├── clear/
    ├── cloudy/
    └── slicing2.py             # Script: Preprocesses the raw SkyPulse samples into 128x128 FPGA-ready tiles.
```

---

##  Pipeline Execution Guide

To replicate the training and hardware export, execute these scripts in order from the **root directory** (`mini-project`):

### Step 1: Prepare the Training Data
Extract the Kaggle 38-Cloud dataset into `sample dataset/`. Run the slicing script to generate the 2,000 balanced tiles.
```cmd
python project\trainer\slicing.py
```
*Output: 2,000 INT8-scaled images deposited into `dataset/clear` and `dataset/cloudy`.*

### Step 2: Prepare the Testing Data (SkyPulse)
Place your raw SkyPulse `.jpg`/`.png` samples into `sample dataset/`. Run the secondary slicing script.
```cmd
python test_data\slicing2.py
```
*Output: 128x128 test tiles deposited into `test_data/clear`. (Note: Manually move the cloudy ones to `test_data/cloudy`).*

### Step 3: Train and Export to Hardware
Run the main PyTorch trainer. This script loads the data, trains the $F=4$ model, calculates FP32 accuracy, and performs the INT8 quantization.
```cmd
python project\trainer\trainer.py
```
*Output: Training metrics printed to terminal. Hardware-ready `conv1_weights.coe` and `fc_weights.coe` are generated inside `project/output files/`.*

### Step 4: Vivado Integration
Open Xilinx Vivado. When generating your Block Memory Generator IPs for the Conv2D and Fully Connected layers, point the initialization parameter to the respective `.coe` files generated in Step 3.