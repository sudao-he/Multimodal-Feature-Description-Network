## Multimodal Feature Description Network
This repository implements the Multimodal Feature Description Network (MFDN) for aligning RGB–IR image pairs in building facade inspection.

This is the first part of the code for our paper:  
[Weakly-aligned Cross-modal Learning Framework for Subsurface Defect Segmentation on Building Facades Using Unmanned Aerial Vehicles](https://www.sciencedirect.com/science/article/abs/pii/S0926580524006824),  
published in ***Automation in Construction***.


## Usage

### 1. Data preparation
- **Project structure**: Place your data under the project root in a directory named after the project, e.g. `UST_Campus`, `Tower_D`, or `Facade_18`.
- **Train / Test split**: Each project directory should contain:
  - **`train/`**: training images (e.g. `train/VIS`, `train/IR`).
  - **`test/`**: testing images and metadata. The `test/` folder must include at least the following **5 subfolders**:
    - `VIS` – visible images
    - `IR` – infrared images
    - `ref` – reference images (if applicable)
    - `homo` – homography (`.mat`) files
    - `landmarks` – landmark (`.mat`) files

### 2. Extract multimodal features
- **Script**: `extract_MMFeat.py`  
- **Purpose**: Run MFDN to extract multimodal keypoints and descriptors for each VIS/IR image pair.
- **Output**: `.features.mat` files are saved under the `features/<Project>/MFDN/` directory.
- **Example**:
  ```bash
  conda env create -f environment.yaml
  conda activate MFDN

  python extract_MMFeat.py \
    --subsets UST_Campus \
    --model ./pretrained/UST_Campus.pth \
    --gpu 0
  ```

### 3. Evaluate the model
- **Script**: `match_ust.py`  
- **Purpose**: Evaluate feature matching performance using the extracted features.
- **Input**: Features under `features/<Project>/MFDN/` and homographies/landmarks under `<Project>/test/`.
- **Example**:
  ```bash
  python match_ust.py \
    --feature_name MFDN \
    --subsets UST_Campus \
    --nums_kp 4096
  ```

### 4. Generate reprojected (warped) images
- **Script**: `reproj_ust.py`  
- **Purpose**: Use the estimated homography to warp VIS images onto IR images (or vice versa) and visualize alignment.
- **Output**: Reprojected images and homography matrices saved under `results/<Project>/...`.
- **Example**:
  ```bash
  python reproj_ust.py \
    --feature_name MFDN \
    --subsets UST_Campus \
    --nums_kp 4096
  ```

### 5. Model training
- **Script**: `train.py`  
- **Purpose**: Train MFDN on RGB–IR pairs.
- **Recommendation**:
  - **Combine** open-source datasets (e.g., [VIS-IR](https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation)) with **your own data** to improve robustness.
  - Adjust `--image_type`, `--datapath`, and training hyperparameters (`--num_epochs`, `--lr`, etc.) as needed.
- **Example**:
  ```bash
  python train.py \
    --image_type VIS_IR \
    --datapath ./UST_Campus/ \
    --gpu 0 \
    --num_epochs 50 \
    --batch_size 2
  ```

## Citation
If you find this work useful in your research or projects, please consider citing the following paper:

```text
@article{he2025weakly,
  title={Weakly-aligned cross-modal learning framework for subsurface defect segmentation on building facades using UAVs},
  author={He, Sudao and Zhao, Gang and Chen, Jun and Zhang, Shenghan and Mishra, Dhanada and Yuen, Matthew Ming-Fai},
  journal={Automation in Construction},
  volume={170},
  pages={105946},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledge
In this project, we use parts of codes in:

[RIFT](https://github.com/LJY-RS/RIFT-multimodal-image-matching)

[RedFeat](https://github.com/ACuOoOoO/ReDFeat)

[SAR-SIFT](https://github.com/yishiliuhuasheng/sar_sift)