import numpy as np
import requests
import tensorflow as tf

from config import MODEL_NAME, NETWORK_NAME, TF_SERVING_PORT


def load_img(path):
    """
    Loads an image from a path and returns it as a numpy array.
    """
    img_ts = tf.keras.preprocessing.image.load_img(path)
    img_np = tf.keras.preprocessing.image.img_to_array(img_ts)
    img_np = np.array([img_np])
    img_np = img_np.astype(np.uint8)
    return img_np


def make_payload(img_np):
    """
    Make tf-serving specific payload.
    """
    return {
        "instances": img_np.tolist()
    }


def make_request(payload):
    """
    Make a request to the tf-serving server.
    """
    url = f"http://{NETWORK_NAME}:{TF_SERVING_PORT}/v1/models/{MODEL_NAME}:predict"
    response = requests.post(url, json=payload)
    return response.json()


def pipeline(src_path):
    """
    Pipeline for pre-processing and making a request to the tf-serving server.
    """
    img_np = load_img(src_path)
    payload = make_payload(img_np)
    response = make_request(payload)
    return img_np, response
