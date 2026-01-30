# Project Proposal  
## Semantic Segmentation of Urban Habitat Types from High-Resolution Aerial Imagery

### 1. Introduction & Motivation

Urban settlement areas are ecologically heterogeneous systems where built structures, sealed surfaces, gardens, trees, and semi-natural elements coexist at very small spatial scales. These habitat structures play an important role for urban biodiversity, microclimate regulation, and ecosystem services, yet they are difficult to monitor continuously. Detailed habitat or biotope mappings are usually created by expert surveys, which are expensive, time-consuming, and updated infrequently.

At the same time, high-resolution aerial imagery is available regularly and consistently for large areas. Recent advances in deep learning for computer vision, especially semantic segmentation models, make it possible to extract fine-grained spatial information from such imagery.

The goal of this project is to investigate whether **urban habitat types can be reliably mapped at pixel level using high-resolution aerial images**, by training a semantic segmentation model on existing biotope mapping data from the City of Zürich. The project aims to demonstrate a scalable, automated approach to derive ecological information in settlement areas, with potential applications in urban planning, environmental monitoring, and ESG-related assessments of real estate portfolios.

---

### 2. Problem Definition

We formulate the task as a **supervised semantic segmentation problem**.

Given a high-resolution aerial image, the model predicts a habitat or biotope class for each pixel. The output is a spatially explicit habitat map aligned with the input image, allowing direct visualization and spatial analysis of ecological structures within urban parcels or neighborhoods.

---

### 3. Data Sources

#### 3.1 Aerial Imagery

The input data consists of high-resolution orthophotos provided by **swisstopo** (e.g. SWISSIMAGE). These images cover the urban area of Zürich at a spatial resolution of approximately 10–25 cm and are available as georeferenced raster datasets in the Swiss coordinate system.

The orthophotos are used as RGB images and serve as the sole input to the deep learning model.

#### 3.2 Habitat / Biotope Mapping

The ground truth labels are derived from the [Biotoptypenkartierung der Stadt Zürich](https://www.stadt-zuerich.ch/geodaten/download/Biotoptypenkartierung_2020?format=10006), which provides a detailed polygon-based mapping of habitat and biotope types in the city. An important simplification for this project is that the habitat polygons do not overlap, which allows direct and unambiguous rasterization.

The original biotope classes are optionally grouped into a smaller number of ecologically meaningful superclasses to reduce label noise and class imbalance.

---

### 4. Dataset Generation

All preprocessing and dataset generation steps are implemented in a Python-based geospatial pipeline.

First, the biotope polygons are rasterized to the same spatial resolution, extent, and coordinate reference system as the aerial imagery. Each pixel in the resulting raster contains an integer class label corresponding to a habitat type.

Next, the aerial imagery raster and the rasterized habitat map are spatially aligned. From these aligned rasters, fixed-size image tiles are extracted. Each tile represents a square area of approximately **100 × 100 meters**, resulting in:

- an image tile (RGB aerial image)
- a corresponding label tile (per-pixel habitat class IDs)

Tiles with insufficient labeled area or missing data are excluded.

To avoid overly optimistic results due to spatial autocorrelation, the dataset is split into training, validation, and test sets using a **spatial split** (e.g. by grid cells or neighborhoods), ensuring that no spatial overlap exists between splits.

---

### 5. Methodology

#### 5.1 Task Formulation

The model learns a mapping from aerial imagery to habitat classes at pixel level:

\[
f_\theta : \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{H \times W \times K}
\]

where \(K\) is the number of habitat classes. A softmax function is applied per pixel, and the predicted class is the argmax over the class probabilities.

#### 5.2 Model Architecture

We use a **semantic segmentation architecture with a pretrained backbone**, implemented in PyTorch.

The general structure consists of:
- a **feature extraction backbone**, pretrained on large image datasets
- a **segmentation decoder** that upsamples feature maps to full image resolution and outputs per-pixel class scores

As a representative example, we propose using a **Transformer-based segmentation model**, such as **SegFormer**, which combines a lightweight Vision Transformer backbone with an efficient decoder head. This architecture is well-suited for capturing both global context and fine-grained spatial patterns in aerial imagery.

#### 5.3 Transfer Learning and Fine-Tuning

The model is initialized with pretrained weights (e.g. ImageNet or remote sensing datasets). Fine-tuning is then performed on the Zürich habitat dataset.

Training details:
- Loss function: pixel-wise cross-entropy loss, optionally with class weighting to address imbalance
- Data augmentation: random rotations, flips, and color jitter to improve generalization
- Optimization: Adam or AdamW optimizer with learning rate scheduling

---

### 6. Evaluation

Model performance is evaluated quantitatively using standard semantic segmentation metrics:
- Pixel accuracy
- Per-class and macro-averaged F1-score
- Intersection-over-Union (IoU)

In addition, qualitative evaluation is performed by visualizing predicted habitat maps and comparing them to the ground truth raster. Predictions are rendered using a fixed color palette, enabling intuitive overlay with the aerial imagery.

---

### 7. Expected Outcomes

The project aims to demonstrate that high-resolution aerial imagery contains sufficient information to predict urban habitat types at pixel level. The expected outcome is a trained semantic segmentation model capable of producing spatially explicit habitat maps for settlement areas.

Beyond the technical contribution, the project illustrates how deep learning can support scalable ecological assessments in urban environments, providing a foundation for applications in urban ecology, planning, and sustainability-oriented real estate analysis.

---

### 8. Limitations and Outlook

The model is trained on expert-derived habitat maps, which may contain uncertainties or mixed habitat types at small scales. Seasonal effects and temporal changes are not explicitly modeled. Future work could extend the approach to multi-temporal imagery, integrate additional data sources such as LiDAR, or refine predictions using hierarchical or multi-scale segmentation strategies.

---

### 9. Conclusion

This project combines geospatial data processing and state-of-the-art deep learning methods to address a real-world problem in urban ecology. By leveraging pretrained semantic segmentation models and high-quality Swiss geodata, it explores the feasibility of automated, fine-grained habitat mapping in settlement areas.
