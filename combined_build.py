import os
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
import matplotlib.colors as mcolors
import transformers

folder = "pictures_raw"

clip = "openai/clip-vit-base-patch32" #or patch16
vit = "google/vit-base-patch16-224-in21k"

# Model setup

model_name = vit

if model_name == clip:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

elif model_name == vit:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)

model.eval()


def get_clip_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        if model_name == clip:
            outputs = model.get_image_features(**inputs)
            print(type(outputs))


        elif model_name == vit:
            outputs = model(**inputs)

        emb = outputs.last_hidden_state[0][0]

    emb = emb.numpy()
    emb = emb / np.linalg.norm(emb)

    return emb


# -----------------------
# Color embedding
# -----------------------
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
# Index images
# -----------------------

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = []
clip_embeddings = []
vit_embeddings = []
color_embeddings = []

for filename in os.listdir(folder):

    ext = os.path.splitext(filename)[1].lower()

    if ext not in valid_ext:
        continue

    path = os.path.join(folder, filename)

    try:
        clip_embeddings.append(get_clip_embedding(path))
        color_embeddings.append(get_color_features(path))
        image_paths.append(path)

        print("Processed:", filename)

    except Exception as e:
        print("Skipping:", filename, e)

clip_embeddings = np.vstack(clip_embeddings)
color_embeddings = np.vstack(color_embeddings)

np.save("vit_embeddings.npy", clip_embeddings)
np.save("color_embeddings.npy", color_embeddings)
np.save("vit_image_paths.npy", np.array(image_paths))

print("Saved index")
print("Images:", len(image_paths))
