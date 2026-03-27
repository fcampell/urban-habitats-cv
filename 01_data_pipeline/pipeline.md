# Zürich Urban Habitat Data Pipeline

This folder contains a modular, high-performance pipeline for generating multi-class training data from the City of Zürich's open geodata. The pipeline follows a **Global Raster Strategy**, prioritizing extraction speed and spatial consistency.

## 1. Spatial Anchoring (Notebook 01)
Instead of random sampling, we use **Building Centroids** (from the Swiss Federal Register of Buildings and Dwellings) as our spatial anchors. 
* **Target:** ~2,000 unique building locations.
* **Format:** A GeoPackage (`.gpkg`) containing unique IDs and train/val splits.
* **Consistency:** All subsequent masks and aerial crops are centered on these exact coordinates.

## 2. The Global Raster Strategy
To avoid redundant vector calculations, the pipeline generates one "Master Mask" for the entire city for each target class.
* **Built-up Area:** Derived from Biotope Mapping (Class 13) in Notebook 02.
* **Grassland:** Derived from Biotope Mapping (Level 2 classes) in Notebook 03.
* **Higher Vegetation:** Derived from the City of Zürich Tree-Height nDSM (Threshold > 2.0m) in Notebook 03.

**Key Advantage:** All global masks are resampled to a unified **0.195m/pixel** resolution, matching the **SWISSIMAGE 10cm** aerial data grid.

## 3. Modular Mask Extraction
A high-speed windowed-reading process "slices" 512x512 pixel patches from the global masks for every anchor.
* **Output:** Binary TIFFs (`0` and `1`) stored in task-specific folders (e.g., `/masks_highveg/`).
* **Performance:** Extraction is near-instantaneous, as it uses pixel-space windows rather than spatial intersections.

## 4. Hierarchical Mask Composition
The final stage of the pipeline merges the modular binary masks into a single **Categorical Ground Truth** image. This allows the model to learn mutually exclusive classes in a single pass.

### Class Hierarchy & Logic:
When masks overlap, the pipeline applies a priority hierarchy:
1.  **Class 3: Higher Vegetation** (Highest Priority - Overlays everything else).
2.  **Class 2: Grassland** (Permeable green surfaces).
3.  **Class 1: Built-up** (Buildings and sealed surfaces).

### 5. Multi-Strategy Modeling Approach
To determine the most efficient and accurate path for urban habitat mapping, the pipeline supports two distinct training strategies:

* **Strategy A: The Modular Ensemble (Hierarchical)** We train three independent, specialized models (one for each binary mask). This allows each model to focus exclusively on the specific texture and spectral features of its target (e.g., the "greenness" of grassland vs. the "height/shadows" of trees). During post-processing, we combine these 1/0 predictions using the **Hierarchical Priority Logic** to resolve overlaps.
    
* **Strategy B: The Unified Multi-Class Model** We train a single SegFormer model using the **Combined 3-Class Mask** as the ground truth. The model learns to classify every pixel into one of the three categories (Grassland, Built-up, High-Veg) in a single inference pass.

**Performance Benchmarking:** Once both strategies are trained, we will perform a comparative analysis focusing on:
**Accuracy (mIoU):** Does a specialized binary model outperform a generalist multi-class model on edge cases (e.g., trees overhanging buildings)?

### Strategy Comparison: Modular vs. Unified

| Feature | **Modular Ensemble** (3 Binary Models) | **Unified Model** (1 Multi-Class Model) |
| :---    | :---                                   | :---                                    |
| **Flexibility** | **High.** You can swap or retrain the "Tree" model if better LiDAR data arrives without touching the "Building" model. | **Low.** Any change to one class (e.g., a new definition of grassland) requires a full retrain of the entire model. |
| **Optimization** | **Specialized.** Each model can use custom loss functions or backbones tuned for its specific texture (e.g., NDVI for grass). | **Generalized.** The model must find a feature set that works for all classes simultaneously, which can lead to compromises. |
| **Inference Speed** | **Slower.** Requires running three separate passes, increasing total compute time and latency. | **Fast.** A single forward pass classifies all pixels at once, making it ideal for large-scale production. |
| **Class Conflict** | **Post-Processed.** Overlaps (e.g., a tree over a roof) are resolved by your manual priority rules. | **Intrinsic.** The model is forced to pick the "most likely" class per pixel during training, handling boundaries naturally. |
| **Data Effort** | **Lower Bar.** You can start training and iterating as soon as a single mask type is ready. | **Higher Bar.** Requires all three mask layers to be perfectly cleaned and merged before training starts. |

### The Bottom Line
* **Modular Ensemble:** Best for **R&D and iterative updates**. Use this if your data sources are updated at different frequencies or if you need maximum precision for a specific category.
* **Unified Model:** Best for **Production and Scale**. Use this if you need an efficient, "all-in-one" deployment that minimizes computational costs during large-scale city mapping.