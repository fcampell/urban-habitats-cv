## 1. Introduction & Motivation

### Core Objective: What are we doing?

The goal of this project is to develop a deep-learning pipeline that automatically converts **high-resolution aerial imagery** into **high-fidelity habitat maps**. Specifically, we are training a semantic segmentation model to recognize and categorize urban surfaces—such as sealed asphalt, private gardens, and tree canopies—at a 10–20 cm pixel level.

We are essentially teaching a computer to "see" and "label" the ecological fabric of the City of Zürich, turning raw photographs into actionable spatial data for urban planning.

### The Problem We are Solving

- **Data Gap**: Existing global datasets (like Sentinel-2) are too coarse ($\geq$ 10m resolution) to see a single tree or a backyard garden.
- **Resource Drain**: High-quality ecological maps currently require "boots on the ground" surveys, which are too expensive to perform every year.
- **The Target**: We aim to bridge this gap by using 10 cm resolution "SWISSIMAGE" orthophotos to provide a continuous, automated update of the city's green infrastructure.

---

### Why Computer Vision?

- **Efficiency**: Automated segmentation reduces the time and resources required for city-wide ecological monitoring by orders of magnitude.
- **Scalability**: Once trained, the model can evaluate thousands of parcels or real estate assets in minutes.
- **The ESG Factor**: Financial institutions and real estate developers face increasing pressure to report on **ESG (Environmental, Social, and Governance)** compliance. This tool provides a data-driven approach to:
    - Assess **Urban Biodiversity** impacts and habitat connectivity.
    - Measure **Green Space Cooling** effects to mitigate Urban Heat Islands.
    - Verify ecological value for portfolio-wide sustainability reporting.

---

## 2. Problem Definition: Semantic Segmentation

### Task Formulation

We define this as a **supervised semantic segmentation problem**. Unlike simple image classification (which labels a whole image), semantic segmentation assigns a specific class label to **every individual pixel**.

### State-of-the-Art (SOTA) Approaches

Modern benchmarks in Remote Sensing favor **Transformer-based architectures** (like SegFormer) over traditional CNNs (like U-Net) because they better capture global context—helping the model understand that "green pixels" near a building are likely a "garden" rather than a "forest."

### Evaluation Metric: mIoU

To measure success, we use **Mean Intersection over Union (mIoU)**:

$$IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union}$$

This metric is more robust than simple accuracy because it penalizes "false positives" and "boundary bleeding," ensuring our predicted habitat maps physically align with reality.

---

## 3. The Evolution of our Approach

### 3.1 Strategy 1: The "Brute Force" Attempt (Initial Prototype)

Our first attempt aimed for maximum granularity using:

- **Data:** 2020 Zürich Biotope Mapping (8 complex superclasses).
- **Input:** 100x100m random tiles from SWISSIMAGE (10cm resolution).
- **Model:** SegFormer-B0.

**The Reality Check:**

- **Complexity:** Distinguishing between 8 ecologically similar classes (e.g., different types of meadows) required more parameters than our hardware could handle.
- **Computational Bottleneck:** Even using Google Colab T4 GPUs, the convergence was slow, and the results showed significant "spectral noise."
- **Outcome:** We realized we needed to simplify the problem to match our computational constraints without losing ecological utility.

---

### 3.2 Strategy 2: The Engineered Modular Approach

We pivoted to a **Hierarchical Pipeline** designed for high performance and modularity. This approach breaks the complex "Urban Habitat" into three distinct, manageable sub-problems.

#### A. The Data Pipeline (Zürich Urban Habitat Pipeline)

1. **Spatial Anchoring:** Instead of random tiles, we anchored our data on **~2,000 Building Centroids** from the Federal Register. This ensures the model learns the specific "settlement area" context we care about.
2. **Global Raster Strategy:** We created "Master Masks" for the entire city at a unified **0.195m/pixel** resolution. This allows for near-instantaneous data extraction.
3. **Modular Extraction:** 512x512 pixel patches are "sliced" from these master masks, training pairs.

#### B. The Hierarchical Logic (Class Priority)

To resolve spatial overlaps (e.g., a tree canopy overhanging a roof), we implemented a priority hierarchy:
1. **Class 3: Higher Vegetation** (Highest Priority - derived from nDSM/Tree-Height data).
2. **Class 2: Grassland** (Permeable green surfaces).
3. **Class 1: Built-up** (Buildings and sealed surfaces).

---

## 4. Modeling Strategies: Modular vs. Unified

We are currently benchmarking two distinct paths to determine the most efficient production workflow:

|**Feature**|**Strategy A: Modular Ensemble**|**Strategy B: Unified Multi-Class**|
|---|---|---|
|**Architecture**|3 Independent Binary Models|1 Single Multi-Class Model|
|**Logic**|Post-processed overlap resolution|Intrinsic (Model decides per pixel)|
|**Pros**|**Maximum Precision:** Each model specializes in one texture (e.g., grass vs. shadows).|**Inference Speed:** One forward pass classifies everything.|
|**Cons**|Slower; requires 3x the compute during inference.|Harder to train; requires perfectly balanced datasets.|
|**Best Use**|Research, R&D, and high-precision mapping.|Scalable production and real-time API tools.|

---

## 5. Preliminary Results & Interpretation


---

## 6. Conclusion & Outlook

### Summary

This project demonstrates that we can transition from manual, expensive surveys to an automated, pixel-level understanding of urban greening. By moving from a "Brute Force" 8-class model to a "Hierarchical Modular" approach, we achieved a more robust and computationally feasible pipeline.

### Next Steps

1. **Final mIoU Benchmarking:** Quantify exactly how much accuracy we gain from the Modular Ensemble vs. the Unified Model.
2. **Multi-Temporal Analysis:** Integrate imagery from different seasons to account for deciduous leaf-off periods.
3. **ESG Dashboard Integration:** Prototype a tool where a user enters a Parcel ID and receives an instant "Ecological Value Score" based on the predicted habitat map.
