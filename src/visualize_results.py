import os
import cv2
import matplotlib.pyplot as plt

from matcher import find_best_matches


query_image = "../data/queries/color_changed.jpg"
gallery_folder = "../data/gallery"

matches = find_best_matches(
    query_image,
    gallery_folder,
    top_k=5
)

os.makedirs("../results", exist_ok=True)

plt.figure(figsize=(15, 8))

# Query image
query = cv2.imread(query_image)
query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

plt.subplot(2, 3, 1)
plt.imshow(query)
plt.title("Query Image")
plt.axis("off")

# Top 5 matches
for index, (filename, similarity) in enumerate(matches):

    image_path = os.path.join(
        gallery_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(2, 3, index + 2)
    plt.imshow(image)

    plt.title(
        f"Rank {index + 1}\n"
        f"{filename}\n"
        f"Similarity: {similarity:.4f}"
    )

    plt.axis("off")


plt.suptitle(
    "Color-Invariant Saree Design Recognition",
    fontsize=16
)

plt.tight_layout()

output_path = "../results/top5_matches.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nResult image saved successfully:")
print(output_path)