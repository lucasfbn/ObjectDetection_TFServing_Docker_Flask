import matplotlib.pyplot as plt
import numpy as np

from config import LABEL_MAP_PATH
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def process_detections(detections):
    """
    Processes the detections to remove unnecessary fields and convert the results to an appropriate np.array
    """

    # In case of batch inference the detections dict contains more than one element. However, we process only one image
    # at a time and are, therefore, only interested in the first element of the list.
    detections = detections["predictions"][0]

    # Pop keys that are either irrelevant or should not get converted to a np array
    detections.pop("raw_detection_scores")
    num_detections = int(detections.pop('num_detections'))

    # Convert to numpy array
    detections = {key: np.array(value[:num_detections])
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections


def _load_label_map():
    return label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH,
                                                              use_display_name=True)


def visualize_detection(path, img_np, detections, label_map):
    """
    Visualizes the detections on the image by drawing boxes and labels.
    """

    img_np = np.squeeze(img_np, axis=0)  # remove batch dimension

    viz_utils.visualize_boxes_and_labels_on_image_array(
        img_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        label_map,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False)

    plt.figure(figsize=(15, 8))
    plt.imshow(img_np)
    plt.tight_layout()
    plt.savefig(path)


def tabularize_detections(label_map, detections):
    """
    Converts the detections (objects and probabilities) to a tabular format.
    """
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    return [
        {"class": label_map[classes[i]]['name'], "score": f"{score * 100:.2f}%"}
        for i, score in enumerate(scores)
        if score >= 0.5
    ]


def pipeline(path, img_np, detections):
    """
    Pipeline for post-processing.
    """
    detections = process_detections(detections)
    label_map = _load_label_map()
    visualize_detection(path, img_np, detections, label_map)
    return tabularize_detections(label_map, detections)
