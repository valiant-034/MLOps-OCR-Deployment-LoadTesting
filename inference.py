import json
import torch
import cv2
import numpy as np

from model import ResnetTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "best.ckpt"

MAPPING_PATH = "mapping.json"


# ---------------------------------------------------
# LOAD MAPPING
# ---------------------------------------------------

with open(MAPPING_PATH, "r") as f:

    mapping = json.load(f)


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

data_config = {

    "input_dims": (
        1,
        128,
        1024,
    ),

    "mapping": mapping,

    "output_dims": (
        256,
    ),
}


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = ResnetTransformer(
    data_config=data_config,
)

checkpoint = torch.load(
    CKPT_PATH,
    map_location=DEVICE,
)

state_dict = checkpoint["state_dict"]

new_state_dict = {}

for k, v in state_dict.items():

    if k.startswith("model."):

        k = k[len("model."):]

    new_state_dict[k] = v

model.load_state_dict(
    new_state_dict
)

model.to(DEVICE)

model.eval()


# ---------------------------------------------------
# TOKEN TO STRING
# ---------------------------------------------------

def tokens_to_text(tokens):

    chars = []

    for t in tokens:

        t = t.item()

        char = mapping[t]

        if char == "<E>":
            break

        if char not in [
            "<S>",
            "<P>",
        ]:

            chars.append(char)

    return "".join(chars)


# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------

def preprocess_image(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    image = cv2.resize(
        image,
        (1024, 128),
    )

    image = image.astype(np.float32)

    image = image / 255.0

    image = torch.tensor(
        image
    ).unsqueeze(0).unsqueeze(0)

    return image


# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

def predict(image):

    x = preprocess_image(image)

    x = x.to(DEVICE)

    with torch.no_grad():

        tokens = model(x)

    text = tokens_to_text(tokens[0])

    return text