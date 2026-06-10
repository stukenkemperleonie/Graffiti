import numpy as np
from PIL import Image
import torch
import matplotlib.colors as mcolors
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

clip = "openai/clip-vit-base-patch32"#16 or 32, preferred 32
vit = "google/vit-base-patch16-224-in21k"

model_name = vit
folder = "smallsize"
pic = "INDIGO_2022-03-13_Z7ii-B_0052_230516T12_10_16"

#INDIGO_2022-03-13_Z7ii-B_0052_230516T12_10_16
#INDIGO_2022-05-16_Z7ii-B_1679_230820T15_11_23
#INDIGO_2022-05-16_Z7ii-B_0077_230820T14_35_28

query = os.path.join(folder, pic + ".png")

if model_name == clip:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model_embeddings = np.load("clip_embeddings.npy")

else:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model_embeddings = np.load("vit_embeddings.npy")

color_embeddings = np.load("color_embeddings.npy")
image_paths = np.load("image_paths.npy")

model.eval()

def get_model_embedding(image_path):
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        if model_name == clip:
            outputs = model.get_image_features(**inputs)

        else:
            outputs = model(**inputs)

        emb = outputs.last_hidden_state[0][0]

    emb = emb.numpy()

    return emb


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

    return hist



def search(query_image, top_k, alpha):


    q_emb = get_model_embedding(query_image).reshape(1, -1)
    q_color = get_color_features(query_image).reshape(1, -1)

    model_sim = cosine_similarity(q_emb, model_embeddings)[0]
    color_sim = cosine_similarity(q_color, color_embeddings)[0]

    final_score = alpha * model_sim + (1 - alpha) * color_sim

    idx = final_score.argsort()[::-1][:top_k] # .argsort() sorts indices of ascending values, [::-1] - [start:stop:step] reverses order (indices of highest values to lowest values), keeps top-k indices

    return [(image_paths[i], final_score[i]) for i in idx]



#results

#results
alphas = [1, 0, 0.5]

for alpha in alphas:
    results = search(query, top_k=10, alpha=alpha)
    results = results[1:] #removes queri from list

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(18, 5))

    fig.suptitle(f"model weight: {alpha:.0%}", fontsize=16)
    axes[0].imshow(Image.open(query))
    axes[0].set_title("QUERY")
    axes[0].axis("off")

    for ax, (path, score) in zip(axes[1:], results):
        ax.imshow(Image.open(path))
        ax.set_title(f"{score:.0%}")
        ax.axis("off")

    plt.tight_layout()
plt.show()
