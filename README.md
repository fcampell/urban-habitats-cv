# Project Proposal  
## Semantic Segmentation of Urban Habitat Types from High-Resolution Aerial Imagery

## 1. Introduction & Motivation

Urban settlement areas are ecologically heterogeneous systems where built structures, sealed surfaces, gardens, trees, and semi-natural elements coexist at very small spatial scales. These habitat structures play an important role for urban biodiversity, microclimate regulation, and ecosystem services, yet they are difficult to monitor continuously. Detailed habitat or biotope mappings are usually created by expert surveys, which are expensive, time-consuming, and suffer from low update frequencies and potential inter-observer variability. >Existing global land-cover datasets lack the spatial resolution (typically $\geq$ 10m) required to capture the fine-grained mosaic of private gardens and green infrastructure. At the same time, high-resolution aerial imagery is available regularly and consistently for large areas. Recent advances in deep learning for computer vision, especially Transformer-based semantic segmentation models, make it possible to extract fine-grained spatial information by moving beyond simple spectral analysis toward contextual scene understanding.

The goal of this project is to investigate whether **urban habitat types can be reliably mapped at pixel level using high-resolution aerial images**, by training a semantic segmentation model on existing biotope mapping data from the City of Zürich. The project aims to demonstrate a scalable, automated approach to derive ecological information in settlement areas, with potential applications in urban planning, environmental monitoring, and ESG-related assessments of real estate portfolios.

---

## 2. Problem Definition

We formulate the task as a **supervised semantic segmentation problem** on high-resolution aerial imagery. The objective is to learn a mapping function that assigns a categorical ecological label to every pixel in a given input scene.

* **Task Formulation:** Given a three-channel input image $X \in \mathbb{R}^{H \times W \times 3}$ (RGB), the model must predict a probability distribution $Y \in \mathbb{R}^{H \times W \times K}$ across $K$ discrete habitat or biotope classes. The final output is a spatially explicit habitat map $M = \text{argmax}(Y)$, where each pixel value $M_{i,j}$ corresponds to a specific ecological category.
* **Spatial Consistency:** Unlike standard image classification, this task requires high-precision localization. The output must maintain an identical spatial extent and coordinate reference system (CRS) as the input, allowing the resulting habitat maps to be directly integrated into Geographic Information Systems (GIS) for neighborhood-scale analysis.
* **Contextual Dependency:** A core challenge addressed in this project is the **spectral ambiguity** of urban surfaces (e.g., distinguishing a "Green Roof" from a "Natural Meadow"). The problem is therefore defined as a **contextual classification task**, where the identity of a pixel depends on its spatial relationship to surrounding structures like buildings, roads, and water bodies.
* **Class Granularity:** The project specifically addresses the gap between coarse "Land Cover" (e.g., "Vegetation") and fine-grained "Habitat Types" (e.g., "Nutrient-poor meadow"), forcing the model to navigate an ecologically meaningful hierarchy.

---

## 3. Data Sources

The project utilizes two primary geospatial datasets to facilitate the training and evaluation of the semantic segmentation model.

### 3.1 Aerial Imagery
The input features consist of high-resolution orthophotos provided by **swisstopo** (e.g., SWISSIMAGE). These images offer a spatial resolution of approximately 10–25 cm, capturing the urban fabric of Zürich with enough detail to distinguish individual trees, garden paths, and small-scale built structures. These georeferenced raster datasets are processed as three-channel RGB images and serve as the primary visual evidence for habitat classification. While these images are restricted to the visible spectrum, their extremely high resolution provides the structural and textural information necessary for the Transformer-based architecture to recognize complex urban patterns through contextual understanding.

### 3.2 Habitat / Biotope Mapping
The ground truth labels are derived from the [Biotoptypenkartierung der Stadt Zürich](https://www.stadt-zuerich.ch/geodaten/download/Biotoptypenkartierung_2020?format=10006), a comprehensive polygon-based inventory of urban habitat and biotope types. This dataset represents the gold standard for ecological mapping in the region, featuring detailed classifications based on expert field surveys. A significant advantage of this dataset is its topological cleanliness; the habitat polygons are non-overlapping, which allows for an unambiguous rasterization process. To ensure the model can generalize effectively despite the high granularity of the original biotope classes, these labels may be selectively grouped into ecologically meaningful superclasses. This grouping strategy addresses potential class imbalances and reduces label noise, creating a more robust target for the supervised learning process while maintaining the ecological integrity of the resulting spatial predictions.

---

## 4. Dataset Generation

The preprocessing and dataset generation workflow is implemented as a high-performance Python-based geospatial pipeline, designed to bridge the gap between vector-based ecological surveys and pixel-based deep learning. The initial phase involves the high-fidelity rasterization of biotope polygons to match the exact spatial resolution, extent, and coordinate reference system of the swisstopo orthophotos. This process assigns an integer class label to each 10 cm pixel, ensuring that the structural nuances of the urban landscape are preserved without spatial offset.

Once the imagery and habitat maps are aligned, the pipeline extracts fixed-size tiles to serve as the fundamental training units. Each tile covers a square area of approximately 100 × 100 meters, a dimension strategically chosen to provide the Transformer backbone with sufficient local context to distinguish between similar vegetation types. This extraction results in paired datasets of RGB aerial image tiles and corresponding label tiles containing per-pixel habitat IDs. To ensure data quality, an automated filtering step excludes tiles that contain missing data or have an insufficient percentage of labeled area.

To address the challenge of spatial autocorrelation—where neighboring pixels and tiles are naturally more similar than distant ones—the dataset is partitioned into training, validation, and test sets using a rigorous spatial split. Rather than a random selection of tiles, the data is divided based on distinct geographic regions or grid-based neighborhoods. This ensures that no spatial overlap exists between the splits, forcing the model to learn generalized ecological features rather than memorizing localized neighborhood patterns.

---

## 5. Methodology

### 5.1 Task Formulation

The core of the methodology involves learning a mapping from high-resolution aerial imagery to habitat classes at a pixel-level. This is formulated as:

$$f_\theta : \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{H \times W \times K}$$

In this expression, $K$ represents the total number of habitat classes. The model outputs a raw score for each class per pixel; subsequently, a softmax function is applied to these scores to produce a probability distribution. The final predicted habitat map is generated by taking the argmax over these class probabilities for every pixel in the spatial grid.

### 5.2 Model Architecture

The project utilizes a semantic segmentation architecture built within the PyTorch framework, leveraging a pretrained backbone for robust feature extraction. The architecture is divided into two primary components: a feature extraction backbone, initialized with weights from large-scale image datasets, and a segmentation decoder designed to upsample low-resolution feature maps back to the original image resolution.

As a representative example of SOTA (State of the Art) performance, we propose the use of **SegFormer**. This Transformer-based model is particularly effective for urban ecology because it eschews fixed positional encodings in favor of a hierarchical Vision Transformer backbone. This allows the model to capture both the global context—essential for understanding the relationship between buildings and green spaces—and the fine-grained spatial patterns required for accurate habitat classification.



### 5.3 Transfer Learning and Fine-Tuning

To optimize performance on specialized geospatial data, the model is initialized with weights pretrained on ImageNet or large-scale remote sensing datasets. Fine-tuning is then conducted on the specific Zürich habitat dataset. The training process incorporates several key strategies:

* **Loss Function:** A pixel-wise cross-entropy loss is employed, with the optional integration of class weighting to mitigate the influence of dominant classes like asphalt or rooftops.
* **Data Augmentation:** To improve generalization and prevent overfitting to specific flight angles or lighting conditions, we apply random rotations, horizontal/vertical flips, and color jittering.
* **Optimization:** The model is trained using the Adam or AdamW optimizer, paired with a learning rate scheduler to ensure stable convergence as the model adapts to the unique spectral characteristics of Swiss urban environments.

---

## 6. Evaluation

The performance of the model is assessed through a dual-lens approach, combining rigorous quantitative metrics with high-fidelity qualitative visualizations to ensure both statistical accuracy and ecological relevance.

### 6.1 Quantitative Metrics
The model is evaluated using standard semantic segmentation benchmarks to capture different aspects of classification performance:
* **Pixel Accuracy:** Provides a baseline measure of the percentage of correctly classified pixels across the entire study area.
* **Per-class and Macro-averaged F1-score:** Measures the balance between precision and recall, ensuring that the model maintains high performance even for rare or spatially restricted habitat types.
* **Intersection-over-Union (IoU):** Serves as the primary metric for measuring the spatial overlap between predicted and ground-truth habitat boundaries, penalizing both false positives and false negatives.

### 6.2 Qualitative Evaluation
In addition to numerical metrics, qualitative evaluation is performed by visualizing the predicted habitat maps and comparing them directly to the ground truth rasters. Predictions are rendered using a fixed, high-contrast color palette, enabling an intuitive overlay with the 10 cm aerial imagery. This visual inspection is critical for identifying spatial artifacts—such as "salt-and-pepper" noise or boundary bleeding—and for validating the model's performance in challenging urban conditions like deep shadows or heterogeneous garden structures.

---

## 7. Expected Outcomes

The primary objective of this project is to demonstrate that high-resolution aerial imagery contains a sufficient density of spectral and structural information to predict complex urban habitat types at the pixel level. The principal technical deliverable will be a trained and validated semantic segmentation model—optimized through a Transformer-based architecture—capable of generating spatially explicit habitat maps for dense settlement areas with high fidelity.

Beyond the immediate technical contribution, the project illustrates the transformative potential of deep learning in supporting scalable ecological assessments. By automating the identification of biotope structures, this work provides a robust foundation for diverse real-world applications:

* **Urban Ecology & Biodiversity:** Enabling city-wide monitoring of habitat connectivity and the identification of ecological "stepping stones" within the built environment.
* **Resilient Urban Planning:** Assisting municipal authorities in assessing the impact of new developments on local green infrastructure and microclimate regulation.
* **ESG & Sustainability Analytics:** Providing real estate investors and portfolio managers with verifiable, spatially explicit data to perform environmental assessments and biodiversity reporting.

Ultimately, this project seeks to prove that automated remote sensing can complement traditional expert surveys, offering a more frequent and cost-effective method for tracking the ecological health of rapidly evolving urban landscapes.

---

## 8. Limitations and Outlook

While this project leverages high-resolution data and modern architectures, several inherent limitations must be acknowledged. The model is trained on expert-derived habitat maps, which may contain historical uncertainties or represent mixed habitat types that are difficult to resolve even at a 10 cm scale. Furthermore, because the current model relies on single-date orthophotos, seasonal variations in vegetation and the impact of transient shadows are not explicitly modeled, which may affect classification consistency across different flight campaigns.

Future work could address these challenges by extending the approach to multi-temporal imagery, allowing the model to learn the phenological signatures of different biotope types. Additionally, the integration of 3D data sources, such as LiDAR-derived Canopy Height Models (CHM), could significantly improve the distinction between spectrally similar but structurally different habitats, such as low-lying meadows versus green roofs or high-canopy trees. These refinements would further move the system toward a fully automated, multi-source ecological monitoring platform.

---

## 9. Conclusion

This project successfully bridges the gap between high-resolution geospatial data processing and state-of-the-art deep learning architectures to address pressing challenges in urban ecology. By leveraging pretrained Transformer-based models and the high-quality geodata provided by the City of Zürich and swisstopo, the research explores the viability of transitioning from manual, resource-intensive surveys to automated, pixel-level habitat mapping. Ultimately, this work demonstrates how computer vision can provide a scalable and repeatable framework for monitoring urban biodiversity, offering essential insights for the future of sustainable urban development and environmental planning.

---

## 10. Acknowledgments

The core conceptualization, motivation, and technical framework of this project were developed by Fadri and Marcel. The refinement of the methodology, structural optimization of the proposal, and technical validation were assisted by Gemini.
