import os

# TF Serving
NETWORK_NAME = os.environ["TF_SERVING_NETWORK_NAME"]
TF_SERVING_PORT = os.environ["TF_SERVING_PORT"]
MODEL_NAME = os.environ["MODEL_NAME"]

# Flask App
ORIGINALS_DIR = "originals"
PROCESSED_DIR = "processed"
LABEL_MAP_PATH = "object_detection/mscoco_label_map.pbtxt"
VALID_FILE_EXTENSIONS = ['png', 'jpg', 'jpeg']
