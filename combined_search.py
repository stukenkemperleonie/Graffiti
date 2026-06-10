import numpy as np
from PIL import Image
import torch
import matplotlib.colors as mcolors
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -----------------------
# Load index
# -----------------------

clip = "openai/clip-vit-base-patch32"#16 or 32, preferred 32
vit = "google/vit-base-patch16-224-in21k"

model_name = vit

pic = "INDIGO_2022-03-13_Z7ii-B_0052_230516T12_10_16"

#INDIGO_2022-03-13_Z7ii-B_0052_230516T12_10_16
#INDIGO_2022-05-16_Z7ii-B_1679_230820T15_11_23
#INDIGO_2022-05-16_Z7ii-B_0077_230820T14_35_28

query = "pictures_raw/" + pic + ".png"

vit_embeddings = np.load("vit_embeddings.npy")
color_embeddings = np.load("color_embeddings.npy")
image_paths = np.load("vit_image_paths.npy")


if model_name == clip:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

else:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
model.eval()

#clip features

def get_clip_embedding(image_path):
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        if model_name == clip:
            outputs = model.get_image_features(**inputs)

        else:
            outputs = model(**inputs)

        emb = outputs.last_hidden_state[0][0]

    emb = emb.numpy()
    emb = emb / np.linalg.norm(emb)

    return emb


# color features
def get_color_features(image_path, bins=(8, 8, 8)):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0

    hsv = mcolors.rgb_to_hsv(img)

    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=bins,
        range=((0, 1), (0, 1), (0, 1))
    )

    hist = hist.flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-8)

    return hist


# -----------------------
# Search
# -----------------------
def search(query_image, top_k=5, alpha=0.5):
    """
    alpha = CLIP weight
    (1 - alpha) = color weight
    """

    q_clip = get_clip_embedding(query_image).reshape(1, -1)
    q_color = get_color_features(query_image).reshape(1, -1)

    clip_sim = cosine_similarity(q_clip, vit_embeddings)[0]
    color_sim = cosine_similarity(q_color, color_embeddings)[0]

    final_score = alpha * clip_sim + (1 - alpha) * color_sim

    idx = final_score.argsort()[::-1][:top_k]

    return [(image_paths[i], final_score[i]) for i in idx]
# -----------------------
# Run example
# -----------------------

alphas_to_test = [0, 1, 0.5]
all_results = {}

for a in alphas_to_test:
    res = search(query, top_k=11, alpha=a)
    all_results[a] = res[1:] #removes query from list

for alpha, results in all_results.items():
    # Determine the title based on alpha
    if alpha == 0:
        title_type = "COLOR ONLY\n(Alpha = 0)"
    elif alpha == 1:
        title_type = "CLIP ONLY\n(Alpha = 1)"
    else:
        title_type = f"MIXED\n(Alpha = {alpha})"

    print(f"\nTop matches for {title_type}:\n")
    for path, score in results:
        print(f"{score:.0%}  {path}")

# -----------------------
# Visualize
# -----------------------
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(18, 5))

    axes[0].imshow(Image.open(query))
    axes[0].set_title(f"QUERY\n({title_type})")
    axes[0].axis("off")

    for ax, (path, score) in zip(axes[1:], results):
        ax.imshow(Image.open(path))
        ax.set_title(f"{score:.0%}")
        ax.axis("off")

    plt.tight_layout()

plt.show()