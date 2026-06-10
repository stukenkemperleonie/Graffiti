import os
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
import matplotlib.colors as mcolors

folder = "smallsize"

clip = "openai/clip-vit-base-patch32" #or patch16
vit = "google/vit-base-patch16-224-in21k"

# Model setup

#choose model (clip or vit)
model_name = vit

if model_name == clip:
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

elif model_name == vit:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)

#evaluation only, no training
model.eval()


def get_model_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    #process images (resize, normalize, reshape arrays,..) to make them "usable" for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad(): #calculation/storing of gradients not necessary for evaluation, speeds up the process
        if model_name == clip:
            outputs = model.get_image_features(**inputs)

        elif model_name == vit:
            outputs = model(**inputs)

        emb = outputs.last_hidden_state[0][0]

    emb = emb.numpy() #768-(vit) or 512 (clip)-feature vector

    return emb


def get_color_features(image_path, bins=(8, 8, 8)):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img) / 255.0

    hsv = mcolors.rgb_to_hsv(img)

    hist, _ = np.histogramdd( #underscore because histogramdd returns the actual histogram counts AND the boundaries for each dimension, which we dont need. The underscore makes sure to only take the first return value (the histogram counts)
        hsv.reshape(-1, 3), #reshape array (image) from (h*w*3) to (rows,3) so that it has 3 values per row (hsv), -1 to automatically calculate number of rows: width*heights = number of pixels = rows
        bins=bins, #hsv each get categorized into 8 "classes"
        range=((0, 1), (0, 1), (0, 1)) #hsv values from 0 to one, for example hue = 0.1 -> roughly red, saturation = 0 -> grey,.., so values are split into 8 bins in a range from 0 to 1
    )

    hist = hist.flatten() #(8,8,8) -> (512,)
    return hist



image_paths = []
model_embeddings = []
color_embeddings = []

for filename in os.listdir(folder):

    path = os.path.join(folder, filename)

    try:
        model_embeddings.append(get_model_embedding(path))
        color_embeddings.append(get_color_features(path))
        image_paths.append(path)

        print("Processed:", filename)

    except Exception as e:
        print("Skipping:", filename, e)

#convert lists to np.ndarrays
model_embeddings = np.vstack(model_embeddings)
color_embeddings = np.vstack(color_embeddings)

if model_name == vit:
    np.save("vit_embeddings.npy", model_embeddings)
elif model_name == clip:
    np.save("clip_embeddings.npy", model_embeddings)

np.save("image_paths.npy", np.array(image_paths))
np.save("color_embeddings.npy", color_embeddings)

print("Saved index")
print("Images:", len(image_paths))
