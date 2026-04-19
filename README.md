# Coarse-to-Fine Multimodal Categorization of Hindi Clinical Conversations

This repository contains a highly optimized, multimodal deep learning architecture designed to classify and analyze medical conversations conducted in Hindi and Hinglish. 

Leveraging a combination of **Text**, **Audio**, and **Visual** data, the system processes patient-practitioner interactions to categorize the underlying communicative purpose across 11 consolidated clinical interaction types. It is specifically engineered to handle the complexities of small-scale, localized datasets (approx. 300 samples) by utilizing advanced regularization and a coarse-to-fine cross-modal fusion strategy.

## ✨ Key Features & Design Choices

* **Coarse-to-Fine Dynamic Attention Fusion (DAF):** The core innovation of this repository. It intelligently gates the integration of all three modalities, moving from global context alignment (coarse) to fine-grained feature fusion.
* **Indic-Optimized Text Encoder:** Utilizes a partially frozen `MuRIL` backbone (`google/muril-base-cased`) to accurately capture the semantic nuances of Hindi and Hinglish text.
* **Fast Audio Processing:** Replaces raw Wav2Vec with Mel-spectrograms processed via a BiGRU network, offering superior stability and speed for short audio clips.
* **Visual Context:** Extracts frame-level features using a pre-trained Vision Transformer (`ViT-base`).
* **Prototype-Aware InfoNCE:** Implements a custom contrastive loss function that compares samples against class prototypes rather than individual instances, drastically improving few-shot learning performance.
* **Smart Class Consolidation:** Maps 29 highly imbalanced raw classes down to 11 well-populated, medically relevant categories to prevent model starvation.
* **Modular Architecture:** Cleanly separated configuration, preprocessing, modeling, and training scripts for easy maintenance and readability.

---

## 📂 Expected Directory Structure

This code is heavily modularized but designed to run seamlessly in **Google Colab**. It expects your Google Drive to be mounted and the project root to contain both your Python scripts and your raw data. Ensure your Drive contains the following structure before running:

```text
/content/drive/MyDrive/Multimodal_Project/
│
├── config.py                # Single source of truth for paths, hyperparams, and class maps
├── preprocess.py            # Converts raw text/audio/video into PyTorch tensors
├── dataset.py               # PyTorch Dataset class and smart splitting logic
├── model.py                 # Core neural network architecture and custom loss functions
├── train.py                 # Training loop, optimizer, and early stopping logic
├── evaluate.py              # Loads best weights and outputs classification metrics
├── requirements.txt         # Required pip packages
│
├── dataset.xlsx             # Main metadata file (file_id, hindi_text, hinglish_text, label)
├── videos/                  # Directory containing raw .mp4 video files
├── audios/                  # Directory containing raw .mp3 or .wav audio files
└── processed_features_v2/   # (Auto-generated) Extracted .pt feature tensors will be saved here
