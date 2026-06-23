# Graffiti Similarity Search System

The goal of this project was to find out whether a visual transformation model can organize a random set of graffiti images and find visually similar graffiti images from this dataset. The aim of the implementation was that, given a single  **"query" image**, the system scans the entire image database and returns the most visually similar images from the dataset.

---

## How It Works

To find the perfect match, the system analyzes photos in two fundamentally different ways: **Vision Transformer Model** and **Color-Based Analysis**.


### 1. Vision Transformer Model
A pre-trained Vision Transformer either **ViT** (google/vit-base-patch16-224-in21k) or **CLIP** (openai/clip-vit-base-patch32) is used to extract high-dimensional feature representations of each image. 
* **How it works:** The `ViTImageProcessor` divides images into fixed-size patches ($16 \times 16$ pixels), which are then linearly projected into embedding vectors (tokens). 
* **The CLS Token:** The final image embedding is taken from the classification token of the last hidden state (`last_hidden_state[0][0]`). This CLS token is prepended to the patch embeddings and serves as the aggregated representation of the entire image.
* **Global Context:** The representation is learned through a self-attention mechanism, which captures global relationships between all image patches simultaneously rather than processing them independently. This allows the model to capture complex visual information such as structure, texture, background context, and artistic graffiti style.
* **Storage:** The resulting 768-dimensional feature vectors are saved into `vit_embeddings.npy` and `vit_image_paths.npy`.

### 2. Color-Based Analysis
As a complementary visual descriptor, color information is analyzed separately.
* **How it works:** Each RGB image is converted into the HSV (Hue, Saturation, Value) color space to better align with human visual perception. 
* **HSV Histogram:** A three-dimensional color histogram is computed over the HSV channels, mapping out the precise color distribution across the image. 
* **Storage:** The histogram array is flattened into a 512-dimensional vector and normalized to ensure clean comparability between images of different sizes. These descriptors are saved into `color_embeddings.npy`.

---

## Similarity Computation & The $\alpha$ Parameter

Once the features are extracted, image similarity is calculated using **cosine similarity**. The system introduces a weighted parameter, $\alpha$ (alpha), to combine and balance the two feature domains:

$$\text{Final Score} = \alpha \cdot \text{ViT Sim} + (1 - \alpha) \cdot \text{Color Sim}$$

When you execute a search query, the system automatically runs and visualizes three separate configurations:
* **Model Only ($\alpha = 1.0$):** Focuses entirely on structural style, background textures, and lettering layout, ignoring the color scheme.
* **Color Only ($\alpha = 0.0$):** Functions strictly as a color matcher, finding images with matching color footprints regardless of the graffiti's shape or style.
* **The Hybrid Combo ($\alpha = 0.5$):** Integrates both local/global context and raw color distribution to find the most accurate visual lookalikes.

---

## File Structure

* `graffiti_embeddings.py` — Preprocesses the image database (`fullsize/` folder), runs the feature extraction pipelines, and saves `vit_embeddings.npy`, `color_embeddings.npy`, and `image_paths.npy`.
* `graffiti_search.py` — Accepts a query image (`query/` folder), computes the weighted cosine similarities, and plots the Top-K results using `matplotlib`.
* `requirements.txt` — Project dependencies (`transformers`, `numpy`, `torch`, `pillow`, `scikit-learn`, `matplotlib`).

---

## Quick Start

1. Install dependencies: `pip3 install -r requirements.txt`
2. Place your gallery images in a `fullsize/` folder and your query image in a `query/` folder.
3. Run the indexer: `python graffiti_embeddings.py`
4. Search and visualize: `python graffiti_search.py`
