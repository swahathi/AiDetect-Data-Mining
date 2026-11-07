# AI-Generated Image Detection 🔍

Detects whether an image is a real camera photograph or AI-generated, and provides human-readable visual and metadata explanations behind every verdict.

---

## Features

- **Metadata Forensics**: Detects camera make/model, lens tags, EXIF timestamps, GPS presence, software signatures, AI generation parameters, and C2PA provenance tags.
- **Visual Statistical Mining**: Measures frequency distribution, GLCM texture features, local binary patterns (LBP), Laplacian sharpness variance, sensor noise residuals, and color saturation dynamics.
- **Explainable Decisions**: Translates model feature weights and deviations into plain-English reasons (e.g. noise uniformity, high-frequency energy, color spread).
- **Production-Ready & Pre-Trained**: Includes pre-trained model bundle (`models/model.joblib`) and evaluation artifacts ready for instant deployment.

---

## Quickstart (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. You can test immediately using sample images from `sample_images/real` or `sample_images/ai`.

---

## Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**. The app will build using `requirements.txt` and run with the pre-trained model in `models/`.

### Option 2: Docker Container
Build and run anywhere (AWS, GCP Cloud Run, Azure, Render, DigitalOcean):

```bash
# Build Docker image
docker build -t ai-image-detector .

# Run container
docker run -p 8501:8501 ai-image-detector
```
Access the application at `http://localhost:8501`.

### Option 3: Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) and choose the **Streamlit** SDK.
2. Push or upload the repo files (including `requirements.txt`, `app.py`, `models/`, and supporting Python scripts).
3. The Space will automatically build and launch.

---

## Retraining the Model

You can train or retrain the classifier on your own dataset or external datasets:

### Using Folder Structure
Organize your images into `real/` and `ai/` directories:
```bash
python train.py --real data/real --ai data/ai
```

### Using a CSV Manifest
```bash
python train.py --data /path/to/dataset --labels-csv /path/to/train.csv --path-col filename --label-col label
```

### Deep Features (Optional CLIP Ensemble)
To train an ensemble combining handcrafted features with pretrained CLIP embeddings (requires PyTorch and open_clip_torch):
```bash
pip install torch open_clip_torch
python train.py --real data/real --ai data/ai --clip
```

---

## Project Structure

```
├── app.py                  # Streamlit web interface
├── predict.py              # Inference pipeline (image -> verdict + reasons)
├── features.py             # Metadata and visual feature extractors
├── explain.py              # Feature contribution and explanation generator
├── data.py                 # Dataset loader and balanced sampling
├── train.py                # Classifier comparison, ablation, and export
├── models/                 # Model bundle and evaluation metrics
│   ├── model.joblib        # Pretrained classification model
│   ├── metrics.json        # Accuracy, F1, AUC, and holdout scores
│   ├── cv_results.csv      # Cross-validation benchmark across models
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
├── sample_images/          # Sample images for testing (real and AI)
├── requirements.txt        # Production dependencies
├── Dockerfile              # Container deployment file
└── .streamlit/             # Streamlit server configuration
```
