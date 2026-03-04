#  Digit Recognition System

Intelligent handwritten digit recognition system using Machine Learning. A neural network trained on the MNIST digits dataset that recognizes hand-drawn numbers (0-9) in real-time through a web interface.

##  Features

-  **Neural Network Model**: MLP (Multi-Layer Perceptron) with 2 hidden layers
-  **97.78% Accuracy**: Trained on 1,797 MNIST digit samples
-  **Real-time Recognition**: Draw and get instant predictions
-  **Streamlit Interface**: Interactive canvas and clean UI
-  **Image Processing**: Automatic binarization, centering, and normalization
-  **Canvas Drawing**: 400x400px drawing area with adjustable brush

##  Requirements

```bash
python >= 3.8
numpy
scikit-learn
streamlit
streamlit-drawable-canvas
pillow
joblib
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

##  Model Details

| Metric | Value |
|--------|-------|
| Algorithm | MLP Classifier |
| Hidden Layers | (128, 64) |
| Input Size | 8x8 pixels (64 features) |
| Output Classes | 10 digits (0-9) |
| Accuracy | 97.78% |
| Dataset Size | 1,797 samples |

##  Technologies

- **Python**
- **scikit-learn**: Machine Learning model
- **Streamlit**: Web interface
- **Pillow**: Image processing
- **numpy**: Numerical computations
- **joblib**: Model persistence

##  Dataset

[MNIST Digits from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- 1,797 samples
- 10 classes (digits 0-9)
- 8x8 pixel images
- Grayscale format

##  Project Structure

```
.
├── app.py                    # Streamlit web application
├── train_model.py           # Model training script
├── modelo_digitos.joblib    # Trained model
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

##  How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Access the web app:**
   - Open `http://localhost:8502` in your browser

4. **Draw and predict:**
   - Draw a digit in the canvas
   - Click "🔍 Adivinar Número" to get the prediction
   - Click "🗑️ Limpiar" to clear the canvas

##  Training the Model

If you want to retrain the model with your own data:

```bash
python train_model.py
```

This will:
- Load the MNIST digits dataset
- Normalize pixel values (0-16 scale)
- Train the neural network
- Save the model as `modelo_digitos.joblib`

##  Image Processing Pipeline

The app automatically processes your drawings:
1. Convert to grayscale
2. Binarize with threshold (127)
3. Detect digit region
4. Resize to 8x8 pixels (maintaining aspect ratio)
5. Center the digit
6. Normalize to 0-16 scale
7. Feed to the model

##  Notes

- The model is trained only once and persisted using `joblib`
- The trained model loads instantly for fast predictions
- Image preprocessing is crucial for accurate recognition
- Drawing style affects prediction accuracy (clear, centered digits work best)
- The virtual environment (`.venv/`) is ignored in Git

## Autor

isra19dev
