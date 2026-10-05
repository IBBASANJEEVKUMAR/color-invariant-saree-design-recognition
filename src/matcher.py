import os
import cv2
import numpy as np

from preprocessing import preprocess_image
from features import extract_features, compare_features


def get_image_features(image_path):
    _, gray_image = preprocess_image(image_path)
    return extract_features(gray_image)


def find_best_matches(query_image, gallery_folder, top_k=5):
    query_features = get_image_features(query_image)

    results = []

    for filename in os.listdir(gallery_folder):

        image_path = os.path.join(
            gallery_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        try:
            gallery_features = get_image_features(
                image_path
            )

            similarity = compare_features(
                query_features,
                gallery_features
            )

            results.append(
                (filename, similarity)
            )

        except Exception:
            continue

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Remove results with exactly the same similarity score
    unique_results = []

    for filename, similarity in results:

        duplicate = False

        for _, selected_similarity in unique_results:

            if abs(
                similarity - selected_similarity
            ) < 0.0001:

                duplicate = True
                break

        if not duplicate:
            unique_results.append(
                (filename, similarity)
            )

        if len(unique_results) == top_k:
            break

    return unique_results


if __name__ == "__main__":

    query_image = "../data/queries/color_changed.jpg"

    gallery_folder = "../data/gallery"

    matches = find_best_matches(
        query_image,
        gallery_folder,
        top_k=5
    )

    print("\nTop 5 matching saree designs:")

    for filename, similarity in matches:

        print(
            f"{filename} -> similarity: {similarity:.4f}"
        )