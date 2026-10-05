# Color-Invariant Saree Design Recognition

## Overview

Color-Invariant Saree Design Recognition is a computer vision project that identifies saree designs based on their visual patterns and textures, even when the saree color changes.

The system takes a query saree image and compares it with a gallery of saree images. It returns the Top 5 most similar designs along with similarity scores.

## Problem Statement

The same saree design can be available in different color combinations.

A recognition system that depends mainly on color may incorrectly treat the same design as different designs.

This project focuses on identifying saree designs using visual information that is less dependent on color, such as:

* Patterns
* Texture
* Edges
* Local visual structure

## How It Works

```text
Input Saree Image
        |
        v
Image Preprocessing
        |
        v
Color Enhancement + Grayscale Conversion
        |
        v
Edge Detection + Texture Extraction
        |
        v
Feature Vector
        |
        v
Cosine Similarity
        |
        v
Top 5 Matching Designs
```

## Dataset

The project uses a collection of 155 saree images.

The images are stored in a gallery folder, while separate query images are used to test the recognition system.

## Feature Extraction

The system combines edge information and texture information.

### Edge Features

Canny edge detection is used to identify important boundaries and structural patterns in the saree image.

### Texture Features

Local Binary Pattern (LBP) based texture histograms are used to capture repeated patterns and texture information.

Combining these features helps the system focus more on the design structure instead of only the saree color.

## Similarity Measurement

Cosine similarity is used to compare the feature vectors of the query image and gallery images.

A higher similarity score means that the two images have more similar visual patterns and textures.

## Example Result

A color-changed version of `0001.jpg` was used as the query image.

The system returned the following Top 5 results:

| Rank | Image    | Similarity |
| ---- | -------- | ---------- |
| 1    | 0001.jpg | 0.9901     |
| 2    | 0081.jpg | 0.9720     |
| 3    | 0089.jpg | 0.9701     |
| 4    | 0079.jpg | 0.9689     |
| 5    | 0004.jpg | 0.9677     |

The original design image `0001.jpg` remained the highest-ranked result even after the query image color was changed.

## Project Structure

```text
SareeDesignAI
│
├── data
│   ├── gallery
│   └── queries
│
├── src
│   ├── preprocessing.py
│   ├── features.py
│   ├── matcher.py
│   ├── create_color_query.py
│   └── visualize_results.py
│
├── results
│   └── top5_matches.png
│
├── notebooks
│
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Computer Vision
* Image Processing
* Local Binary Pattern (LBP)
* Cosine Similarity

## How to Run

### 1. Open the project folder

Open the `SareeDesignAI` folder in VS Code.

### 2. Activate the virtual environment

Open the VS Code terminal and run:

```text
.venv\Scripts\Activate.ps1
```

### 3. Run the matcher

```text
cd src
python matcher.py
```

### 4. Generate the visual result

```text
python visualize_results.py
```

The result will be saved as:

```text
results/top5_matches.png
```

## Main Project Files

### `preprocessing.py`

Prepares the input image by resizing it, enhancing the image using LAB color processing, and converting it into grayscale.

### `features.py`

Extracts edge and texture information using Canny edge detection and Local Binary Pattern (LBP).

### `matcher.py`

Compares the query image with the gallery images using cosine similarity and returns the Top 5 matches.

### `create_color_query.py`

Creates a color-changed version of a gallery image to test whether the system can recognize the same design despite a color change.

### `visualize_results.py`

Creates a visual comparison showing the query image and the Top 5 matching saree designs.

## Results

The system successfully identified the original design as the highest-ranked match after changing the color of the query image.

This demonstrates that the extracted texture and structural features can help recognize a saree design while reducing the effect of color variation.

## Future Improvements

The project can be improved further by:

* Using deep learning image embeddings
* Adding more saree images
* Testing different color transformations
* Using advanced texture descriptors
* Building a web interface for uploading saree images
* Supporting real-time similarity search
* Evaluating the system on a larger test dataset

## Conclusion

This project demonstrates a computer vision approach for recognizing saree designs while reducing the effect of color variations.

The system combines image preprocessing, edge detection, texture analysis and cosine similarity to retrieve visually similar saree designs.

The approach provides a simple and interpretable baseline for color-invariant saree design recognition.

```
