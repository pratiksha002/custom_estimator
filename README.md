# Smart Preprocessor (Custom Estimator)

## 🚀 Overview
This project implements a custom scikit-learn compatible preprocessing estimator that automates data preprocessing.

## 🔥 Features
- Automatic numerical & categorical detection
- Missing value handling
- One-hot encoding
- Feature scaling
- Polynomial feature generation
- Outlier handling
- Pipeline compatible

## 🧠 Usage

```python
from src.preprocessor import SmartPreprocessor

preprocessor = SmartPreprocessor(add_polynomial=True)
X_transformed = preprocessor.fit_transform(X)

## ⚙️ Pipeline Example
Pipeline([
    ("preprocessing", SmartPreprocessor()),
    ("model", Ridge())
])
##📦 Installation
pip install -r requirements.txt
📊 Future Improvements
Target encoding
Auto feature selection
Logging system
AutoML integration


---

# 🧪 5. (Optional) `notebook.ipynb`

Just use it to:
- Load real dataset
- Test transformations
- Visualize features

---

# ✅ What You Now Have

You built:
- ✅ Custom sklearn-compatible estimator  
- ✅ Pipeline integration  
- ✅ Feature engineering system  
- ✅ GitHub-ready project  

---

# 🎯 Next Step (VERY IMPORTANT)

Now do this:

```bash
git add .
git commit -m "Added SmartPreprocessor and training pipeline"
git push