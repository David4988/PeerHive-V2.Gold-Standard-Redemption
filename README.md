# PeerHive: Multimodal Burnout Detection

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-archived-red.svg)

A rapid-prototype project from a 1-week sprint focused on multimodal (text + image) burnout detection. This repository contains the code used to fuse text and image sentiment to map user-generated content to distinct burnout zones, forming the basis of research presented in an IEEE publication.

## 🚀 Project Goal

The primary objective of this project was to explore the fusion of text and image data to create a more accurate and nuanced model for detecting early signs of burnout. We aimed to move beyond simple positive/negative sentiment and map complex emotional expressions (from platforms like GoEmotions, Flickr30k, and Memotion) to a practical burnout framework.

## 💡 Methodology

Our approach consisted of a two-stage process: defining a practical burnout framework and implementing a multimodal fusion model to classify content.

### 1. Burnout Zone Framework

We mapped fine-grained emotions (derived from text and images) to a four-tier burnout-risk framework:

* **Calm:** Mapped from emotions like *Joy, Admiration, Approval*.
* **Stressed:** Mapped from emotions like *Anger, Annoyance, Disapproval*.
* **Overwhelmed:** Mapped from emotions like *Sadness, Fear, Grief, Confusion*.
* **Burned Out:** Mapped from emotions like *Despair, Disgust, Exhaustion*.

### 2. Multimodal Fusion Model

To classify content into the framework above, we processed text and image data separately before fusing them.

1.  **Text Processing:** All captions and text snippets were processed using a pre-trained `DistilBERT` model from the Hugging Face Transformers library to extract contextual text embeddings.
2.  **Image Processing:** Corresponding images were processed using `CLIP` to extract powerful multimodal (image-text) embeddings.
3.  **Early Fusion:** The resulting text embeddings and image embeddings were concatenated into a single, unified feature vector.
4.  **Classification:** This fused vector was fed into a Multi-Layer Perceptron (MLP) head, which was then trained to classify the (text, image) pair into one of the four burnout zones.

## 🛠️ Tech Stack

This project leverages a combination of deep learning models and data processing libraries:

* **Python 3.9+**
* **Transformers (Hugging Face):** For text analysis, primarily using `DistilBERT`.
* **CLIP:** For generating joint text-image embeddings and sentiment.
* **PyTorch / TensorFlow:** As the deep learning framework.
* **Pandas:** For data manipulation and schema management.
* **Scikit-learn:** For model evaluation (Accuracy, F1-Score, Confusion Matrix).
* **Matplotlib / Seaborn:** For generating evaluation plots and heatmaps.

## Ppt - [Final ppt](https://docs.google.com/presentation/d/1tzBVQsPU5q9zK5zuK5FZOoNLeLdnMNi9/edit?usp=drive_link&ouid=100773914437128373863&rtpof=true&sd=true)
## Video Demo - [PeerHive]()

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.9 or higher
* `pip` (Python package installer)

### Installation

1.  **Clone the repo:**
    ```sh
    git clone https://github.com/David4988/PeerHive-V2.Gold-Standard-Redemption.git
    cd PeerHive
    ```

2.  **Install requirements:**
    This project includes a `requirements.txt` file to install all necessary dependencies.
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the main script (example):**
    *(You can update this command to match your main execution file)*
    ```sh
    python main.py
    ```

## 📊 Evaluation

The model's performance was evaluated against a held-out test set using several standard classification metrics. The primary goal was to achieve a high F1-score, given the potential for class imbalance in real-world burnout data.

* **Accuracy**
* **Precision, Recall, and F1-Score (Weighted)**
* **Confusion Matrix** (Generated to analyze misclassification patterns between zones, e.g., 'Stressed' vs. 'Overwhelmed').
* **Per-Class F1-Score Bar Charts**

*[See the final paper for detailed results, plots, and analysis.]*

## 📊 Datasets

This project's models were trained and evaluated on a combination of three key datasets:
* **GoEmotions 2.0:** Used for its fine-grained text emotion labels.
* **Flickr30k:** Provided a rich source of image-caption pairs.
* **Memotion:** Offered a challenging dataset of multimodal meme content.

## 🚧 Limitations & Future Work

Given the 1-week sprint constraint, this project has several limitations that provide clear avenues for future research:

* **Limitations:**
    * The burnout mapping is an approximation and could be refined with domain-expert (psychologist) input.
    * The model was trained on public datasets, which may not perfectly represent real-world private user content.
    * The fusion model (early-stage concatenation) is relatively simple.

* **Future Work:**
    * Implement more advanced fusion techniques (e.g., late fusion, cross-attention transformers).
    * Expand the dataset with more diverse and realistic examples of workplace or academic stress.
    * Deploy the model as part of a real-time wellness-support tool (with appropriate ethical and privacy safeguards).

## 📄 License

Distributed under the MIT License. See `LICENSE.txt` for more information.


