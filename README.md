# Semantic Segmentation of Urban Habitat Types: The Zürich Case Study

This repository contains an end-to-end pipeline for mapping urban biodiversity and land cover at a 10cm resolution. By leveraging **SegFormer (B3)** transformers and the high-quality open geodata of the **City of Zürich**, we transition from manual, resource-intensive biotope surveys to automated, pixel-level habitat classification.

## 1\. Project Vision

Urban ecosystems are a complex mosaic of sealed surfaces, private gardens, and tree canopies. Standard global land-cover datasets (10m+) are too coarse to capture these fine-grained structures. This project investigates whether **Transformer-based models** can reliably map these habitats using high-resolution aerial imagery, providing a scalable framework for urban ecological monitoring.

## 2\. Data & Modeling Pipeline

The project follows a **Global Raster Strategy**, optimizing for spatial consistency and high-speed data extraction.

### Pipeline Flowchart

![flowchart](https://github.com/fcampell/urban-habitats-cv/blob/main/images/flowchart.jpg)

1.  **Spatial Anchoring:** We used 10,000 building centroids from the Swiss Federal Register (GWR/GWZ) as coordinates to sample $100m \times 100m$ patches.
2.  **Global Rasterization:** Instead of cropping vectors, we rasterized the entire city's biotope and tree-height data into a unified 10cm grid.
3.  **Modular Extraction:** We "sliced" 512x512 pixel patches (JPG for aerials, PNG for masks) for three binary tasks: Built-Up, Grassland, and High Vegetation.
4.  **Hierarchical Synthesis:** To create a multi-class ground truth, we applied a "Painter's Algorithm" priority: **High Vegetation \> Grassland \> Built-Up \> Other**.
5.  **Modeling:** We fine-tuned individual **SegFormer-B3** models for modular tasks and compared them against a unified Multi-Class model and a Hierarchical Ensemble.

## 3\. Data Sources

To replicate this study, the following raw datasets are required:

| Dataset | Source | Description |
| :--- | :--- | :--- |
| **SWISSIMAGE 10cm** | [swisstopo WMS](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html) | High-resolution RGB aerial imagery. |
| **Biotoptypen 2020** | [Stadt Zürich Open Data](https://data.stadt-zuerich.ch/dataset/geo_biotoptypenkartierung) | Expert-surveyed habitat maps (Vector). |
| **Tree Height Model** | [Stadt Zürich nDSM](https://data.stadt-zuerich.ch/dataset/geo_baumhoehen_2022__chm_aus_lidar_) | LiDAR-derived normalized Digital Surface Model. |
| **GWR / GWZ** | [Federal Register](https://data.stadt-zuerich.ch/dataset/geo_gebaeude__und_wohnungsregister_der_stadt_zuerich__gwz__gemaess_gwr_datenmodell) | Building and Dwelling Register for spatial anchors. |

## 4\. Results & Inference

Performance was evaluated on a 15% hold-out test set (1,500 images). We measured success using the **Mean Intersection over Union (mIoU)**, which penalizes false positives and rewards precise overlap.

### Test Set Performance

| Segmentation Task | Model Architecture | Average mIoU |
| :--- | :--- | :--- |
| **Built-Up Area** | SegFormer B3 (Fine-tuned) | **0.74** |
| **Grassland** | SegFormer B3 (Fine-tuned) | **0.61** |
| **Bushes & Trees** | SegFormer B3 (Fine-tuned) | **0.73** |
| **Multi-Class** | SegFormer B3 (Fine-tuned) | **0.53** |
| **Multi-Class** | Ensemble (Modular 1-3) | **0.51** |

### Key Findings

  * **Modular Strength:** Individual binary models (especially Built-up and Trees) show high robustness. The higher score for binary models vs. multi-class suggests that specialized feature extraction is beneficial for distinct textures like canopy shadows.
  * **The "Shadow" Challenge:** The slightly lower score for Grassland (0.61) is largely attributed to spectral similarities between shadows on grass and shadows on asphalt.
  * **Unified vs. Ensemble:** The fine-tuned multi-class model slightly outperformed the hierarchical ensemble (0.53 vs 0.51), indicating that the model is capable of learning class boundaries internally better than a fixed manual priority rule. Altough in some samples where there was a high class imbalance (only built up area with small patches of green), the rule based ensemble predictions performed better, as the model focussing only on segmenting higher vegetation found these patches with better accuracy than the multi-class model.


## 5\. Repository Structure

```text
├── 01_data_pipeline/     # Notebooks for WMS download and rasterization, Modular mask generation (Built-up, Grass, Trees)
├── 02_model_pipeline/    # SegFormer-B3 training notebooks (Colab/Local), Ensemble logic and test-set evaluation
└── archive/              # prototypes and old data pipeline work
```

## 6\. How to Use

1.  **Data Preparation Pipeline:** Run all notebooks in `01_data_pipeline`
2.  **Training:** Run notebooks in `02_model_pipeline`
3.  **Evaluate:** Run `05_inference_test.ipynb` in `02_model_pipeline` to generate final metrics on the test set.

The model pipeline is built to run on Google Collab to use GPU's for training. With the suggested sample size of about 
7'500 training samples, most of the models take about 2 hours to train 10 epochs on a `A100 GPU`.

## 7\. Reference

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS.

-----

**Authors:** Fadri Campell & Marcel Amrein

**Technical Assistance:** Assisted by Gemini

**License:** MIT
