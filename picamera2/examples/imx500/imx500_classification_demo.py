import argparse
import sys
import time
from typing import List

import cv2
import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

last_detections = []
LABELS = None


class Classification:
    def __init__(self, idx: int, score: float):
        """Create a Classification object, recording the idx and score."""
        self.idx = idx
        self.score = score


def get_label(request: CompletedRequest, idx: int) -> str:
    """Retrieve the label corresponding to the classification index."""
    global LABELS
    if LABELS is None:
        LABELS = intrinsics.labels
        output_shapes = imx500.get_output_shapes(request.get_metadata())
        if output_shapes:
            output_tensor_size = output_shapes[0][0]
            # If labels include a leading background and output omits it, drop first label.
            if len(LABELS) == output_tensor_size + 1:
                LABELS = LABELS[1:]
    return LABELS[idx]


def parse_and_draw_classification_results(request: CompletedRequest):
    """Analyse and draw the classification results in the output tensor."""
    results = parse_classification_results(request)
    draw_classification_results(request, results)


def parse_classification_results(request: CompletedRequest) -> List[Classification]:
    """Parse the output tensor into the classification results above the threshold."""
    global last_detections
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        return last_detections
    np_output = np_outputs[0]
    if intrinsics.softmax:
        np_output = softmax(np_output)
    k = min(3, np_output.shape[0])
    top_indices = np.argpartition(-np_output, k-1)[:k]
    top_indices = top_indices[np.argsort(-np_output[top_indices])]  # Sort by score
    last_detections = [Classification(index, np_output[index]) for index in top_indices]
    return last_detections


def draw_classification_results(request: CompletedRequest, results: List[Classification], stream: str = "main"):
    """Draw the classification results for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        if intrinsics.preserve_aspect_ratio:
            # Drawing ROI box
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
            text_left, text_top = b_x, b_y + 20
        else:
            text_left, text_top = 0, 0
        # Drawing labels (in the ROI box if it exists)
        for index, result in enumerate(results):
            label = get_label(request, idx=result.idx)
            text = f"{label}: {result.score:.3f}"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = text_left + 5
            text_y = text_top + 15 + index * 20

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.3
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("-s", "--softmax", action=argparse.BooleanOptionalAction, help="Add post-process softmax")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preprocess the image with preserve aspect ratio")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "classification"
        # Ensure the inference ROI matches our 224x224 model input even if
        # the RPK lacks intrinsics metadata.
        imx500.config['input_tensor_size'] = (224, 224)
        imx500.set_inference_aspect_ratio((224, 224))
    elif intrinsics.task != "classification":
        print("Network is not a classification task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/imagenet_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    # Register the callback to parse and draw classification results
    picam2.pre_callback = parse_and_draw_classification_results

    while True:
        time.sleep(0.5)
