# Deep Learning for Semantic Segmentation of Urban Biotopes

## Overview
This project aims to automate the ecological mapping of the city of Zurich using high-resolution (10cm) aerial imagery and deep learning. By training a Vision Transformer (SegFormer) on SWISSIMAGE orthophotos and ground-truth vector data from the Stadt Zurich open data portal, the system segments urban landscapes into 8 distinct ecological superclasses (e.g., Impervious Surfaces, Forest Canopy, Ecologically Valuable Meadows).

The original project proposal and methodology can be found in `proposal.md`.

---

## Current Project Status: Prototype Phase Completed

The project is currently transitioning from the prototyping phase to the production phase. We have successfully developed a complete end-to-end pipeline that transforms raw GIS vector data into a functioning PyTorch deep learning model. 

The codebase currently consists of three sequential Jupyter Notebooks, which directly fulfill the core requirements outlined in the project proposal:

### 1. Exploratory Data Analysis & Preparation (`01_eda_and_preparation.ipynb`)
**Aligns with Proposal Section 4.1 (Vector Data Preparation)**
* Loaded and analyzed the raw biotope shapefiles provided by the city of Zurich.
* Filtered out overlapping or invalid polygons to ensure data integrity.
* Successfully mapped dozens of sub-categories into the 8 target ecological superclasses to create a balanced, machine-learning-ready taxonomy.
* Exported the cleaned data as a standardized GeoPackage.

### 2. Rasterization & Tiling Pipeline (`02_rasterization_pipeline.ipynb`)
**Aligns with Proposal Section 4.2 (Rasterization & Tiling)**
* Developed a geospatial pipeline using `rasterio` and `geopandas` to align the vector data with high-resolution RGB aerial imagery (SWISSIMAGE).
* Burned the vector polygons into exact pixel masks matching the 10cm resolution of the imagery.
* Sliced the massive raster files into smaller, manageable 1000x1000 pixel tiles (representing 100x100m real-world areas) suitable for GPU memory constraints.

### 3. Deep Learning Prototype (`03_prototype.ipynb`)
**Aligns with Proposal Section 5 (Deep Learning Architecture & Training) and Section 6 (Evaluation)**
This notebook serves as the fully functional, optimized prototype of the neural network.
* **Data Pipeline:** Built a custom PyTorch `Dataset` and `DataLoader`. Utilized `Albumentations` for data augmentation (flips, rotations, color jitter) and applied strict ImageNet mathematical normalization.
* **Model Architecture:** Initialized Hugging Face's `SegFormer` (nvidia/mit-b0), dynamically replacing the pre-trained classification head with a custom 8-class head.
* **Hardware Optimization:** Resized inputs to 1024x1024 to ensure perfect divisibility by 32, bypassing memory stride fragmentation issues on Apple Silicon (MPS) backends.
* **Training Mechanics:** * Implemented an 80/20 Train/Validation split to monitor generalization.
    * Utilized **Focal Loss** combined with inverse class weighting to force the model to learn rare ecological classes (like Water or Agriculture) rather than over-predicting dominant classes (like Impervious surfaces).
    * Integrated a `ReduceLROnPlateau` Learning Rate Scheduler to aid in mathematical convergence.
    * Implemented an Early Stopping mechanism to halt training and save the optimal model weights when the accuracy metric plateaus.
* **Evaluation:** Calculated the strict Mean Intersection-over-Union (mIoU) metric at the end of each epoch and built a qualitative visualization tool to compare original imagery, ground truth masks, and model predictions side-by-side.

---

## Next Steps: Production & Scaling

With the prototype mathematically validated and the model architecture proven to learn the spatial relationships of the biotope classes, the next phase of the project involves scaling to production. 

The upcoming developments will include:
1.  **Dataset Scaling:** Automating the download and processing of the remaining SWISSIMAGE tiles to cover the entire spatial extent of Zurich.
2.  **Rigorous Data Splitting:** Implementing a strict, spatially separated Train/Validation/Test split across different geographical zones of the city to prevent data leakage.
3.  **Production Training Run:** Training the model on the full, thousands-of-images dataset to drive the mIoU accuracy up to production standards.
4.  **Full City Inference:** Deploying the trained model to generate a continuous, seamless ecological raster map of the entire city.