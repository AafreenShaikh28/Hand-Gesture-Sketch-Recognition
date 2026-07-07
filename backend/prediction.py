import tensorflow as tf
import cv2
import numpy as np
import dotenv
import os

dotenv.load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

# Loaded once at import time instead of on every request — loading the
# model inside predict() (as in the original code) means every single
# prediction re-reads the .keras file from disk and rebuilds the graph,
# which is slow and unnecessary since the model never changes.
_model = tf.keras.models.load_model(MODEL_PATH)


# =============================================================================
# Helper functions
# =============================================================================

def _threshold_and_orient(gray: np.ndarray) -> np.ndarray:
    """
    Step 3/4: Threshold into black/white, then make sure the digit ends up
    WHITE (255) on a BLACK (0) background, matching MNIST's convention.

    We use Otsu's method so the threshold adapts to the actual contrast of
    each frame instead of a fixed magic number. Otsu doesn't know which side
    is "foreground", so we check which class covers less area — the digit
    (foreground) is always the minority of pixels, the background covers
    most of the canvas.
    """
    # THRESH_BINARY: pixels > threshold -> 255, else 0. Otsu picks the
    # threshold value automatically from the image histogram.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixel_count = np.count_nonzero(binary == 255)
    black_pixel_count = binary.size - white_pixel_count

    # If white pixels are the majority, the digit is currently black-on-white
    # (e.g. a white canvas with a dark pen stroke) — invert so digit = white.
    if white_pixel_count > black_pixel_count:
        binary = cv2.bitwise_not(binary)

    return binary


def _bounding_box(binary: np.ndarray):
    """
    Step 5: Find the bounding box of the drawn (white) pixels.
    Returns (x, y, w, h), or None if the image is blank.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    return cv2.boundingRect(coords)  # (x, y, w, h)


def _resize_preserving_aspect(digit: np.ndarray, target_size: int = 20) -> np.ndarray:
    """
    Step 8: Resize the cropped digit so its LONGER side becomes
    `target_size` pixels, preserving aspect ratio (no squashing/stretching).

    MNIST digits are normalized to fit in a 20x20 box within the final
    28x28 image (the remaining 4px margin is what centering fills in),
    so we mirror that convention here.
    """
    h, w = digit.shape
    if h > w:
        new_h = target_size
        new_w = max(1, round(w * (target_size / h)))
    else:
        new_w = target_size
        new_h = max(1, round(h * (target_size / w)))

    # INTER_AREA is the recommended interpolation for shrinking images —
    # it avoids the aliasing/noise that INTER_LINEAR can introduce on
    # already-thin, high-contrast pen strokes.
    return cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _center_in_canvas(digit: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    """
    Step 9: Place the resized digit in the middle of a canvas_size x canvas_size
    black canvas, so the digit's centroid roughly lines up with the center
    of the final image, just like real MNIST samples.
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    h, w = digit.shape

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = digit
    return canvas


# =============================================================================
# Main preprocessing pipeline
# =============================================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw PNG bytes from the frontend into a (28, 28, 1) float32
    array normalized to [0, 1], preprocessed to closely mimic the MNIST
    dataset's own preprocessing conventions.
    """
    # Step 1: Decode image bytes into a numpy array OpenCV can read.
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Step 2: Decode directly to grayscale (single channel), since color
    # information isn't meaningful for a black/white digit classifier.
    gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("preprocess_image: failed to decode image bytes")

    # Steps 3-4: Threshold to pure black/white, oriented digit=white/bg=black.
    binary = _threshold_and_orient(gray)

    # Step 5: Locate the bounding box of the drawn digit.
    bbox = _bounding_box(binary)

    if bbox is None:
        # Nothing drawn (blank canvas) — return an all-black frame rather
        # than crashing; predict() will just get a low-confidence result.
        empty = np.zeros((28, 28, 1), dtype=np.float32)
        return empty

    x, y, w, h = bbox

    # Step 6: Crop tightly around the digit using the bounding box.
    cropped = binary[y:y + h, x:x + w]

    # Step 7: Add a small padding border around the digit before resizing.
    # This keeps a bit of breathing room, mirroring the fact that MNIST
    # digits rarely touch their own bounding box edges.
    pad = max(2, int(0.1 * max(w, h)))
    padded = cv2.copyMakeBorder(
        cropped, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,  # black, matching our black background convention
    )

    # Step 8: Resize preserving aspect ratio (longest side -> 20px, the
    # standard MNIST convention, leaving room for centering below).
    resized = _resize_preserving_aspect(padded, target_size=20)

    # Step 9: Center the resized digit inside the final 28x28 canvas.
    centered = _center_in_canvas(resized, canvas_size=28)

    # Step 10: Normalize pixel values from [0, 255] to [0, 1] float32,
    # matching the scale the model was trained on.
    normalized = centered.astype(np.float32) / 255.0

    # Step 11: Reshape to (28, 28, 1) — channel-last, single grayscale channel.
    return normalized.reshape(28, 28, 1)


def predict(image_array: np.ndarray):
    """
    Runs the preprocessed (28, 28, 1) image through the model and returns
    the top-3 predicted digits with their confidences.
    """
    predictions = _model.predict(
        np.expand_dims(image_array, axis=0),
        verbose=0,
    )[0]

    top_indices = np.argsort(predictions)[::-1][:3]
    guesses = [
        {
            "label": str(i),
            "confidence": float(predictions[i]),
        }
        for i in top_indices
    ]
    return guesses