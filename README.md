# AI-Generated Image Detection

Detects whether an image is a real photo or AI-generated, and explains why.

**1. Install the libraries**
```bash
pip install -r requirements.txt
```

**2. Download the Kaggle dataset `malaxmels/ai-detection-dataset` and unzip it so you have a `real` and an `ai` folder**
```bash
kaggle datasets download -d malaxmels/ai-detection-dataset --unzip -p archive
```

**3. Train the model**
```bash
python train.py --data archive/dataset --max-per-class 3000
```

**4. (Optional) Train with CLIP for higher accuracy**
```bash
python train.py --data archive/dataset --max-per-class 3000 --clip
```

**5. Start the web app**
```bash
streamlit run app.py
```

**6. Open the link it prints and upload an image**
```
http://localhost:8501
```

**If something fails**

`EmptyDataError`:
```bash
del models/features*.csv
```
