import cv2
import numpy as np


def extract_features(gray_image):
    # Resize image
    image = cv2.resize(gray_image, (128, 128))

    # Improve contrast
    image = cv2.equalizeHist(image)

    # Edge information
    edges = cv2.Canny(
        image,
        threshold1=40,
        threshold2=120
    )

    # Texture information using Local Binary Pattern
    lbp = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):

            center = image[y, x]

            binary_value = 0

            binary_value |= (image[y - 1, x - 1] >= center) << 7
            binary_value |= (image[y - 1, x] >= center) << 6
            binary_value |= (image[y - 1, x + 1] >= center) << 5
            binary_value |= (image[y, x + 1] >= center) << 4
            binary_value |= (image[y + 1, x + 1] >= center) << 3
            binary_value |= (image[y + 1, x] >= center) << 2
            binary_value |= (image[y + 1, x - 1] >= center) << 1
            binary_value |= (image[y, x - 1] >= center)

            lbp[y, x] = binary_value

    # Divide image into blocks and create texture histograms
    texture_features = []

    block_size = 32

    for y in range(0, 128, block_size):
        for x in range(0, 128, block_size):

            block = lbp[
                y:y + block_size,
                x:x + block_size
            ]

            histogram = cv2.calcHist(
                [block],
                [0],
                None,
                [256],
                [0, 256]
            )

            histogram = histogram.flatten()

            histogram = histogram / (
                np.sum(histogram) + 1e-7
            )

            texture_features.extend(histogram)

    # Edge histogram
    edge_histogram = cv2.calcHist(
        [edges],
        [0],
        None,
        [32],
        [0, 256]
    ).flatten()

    edge_histogram = edge_histogram / (
        np.sum(edge_histogram) + 1e-7
    )

    # Combine texture and edge features
    features = np.concatenate(
        [
            np.array(texture_features),
            edge_histogram
        ]
    )

    return features.astype(np.float32)


def compare_features(features1, features2):
    features1 = np.array(features1)
    features2 = np.array(features2)

    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(
        features1,
        features2
    ) / (norm1 * norm2)

    return float(similarity)


if __name__ == "__main__":
    print("Advanced saree feature extraction module is ready.")