# SVM-Based Image Classifier (Cats vs Dogs)

This project uses a Support Vector Machine (SVM) to classify images of **cats and dogs. It processes image data, trains a model, and evaluates its performance using standard metrics.

## Features

- Image loading and preprocessing
- Train/test data split
- SVM model with linear kernel
- Evaluation using accuracy and classification report
- Color-coded terminal output using `colorama`
- Progress bar with `tqdm`

## Tech Stack
- Python 3
- scikit-learn
- NumPy
- tqdm
- colorama
- matplotlib (optional - for image visualization)
## Folder Structure

main.py # Main script
├── utils.py # Image loading and preprocessing logic
├── train/
│ └── PetImages/ # Folder with Cat and Dog image subfolders
├── svm_model.pkl # (optional) Saved model
└── README.md # Project documentation

##  Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/svm-cats-vs-dogs.git
   cd svm-cats-vs-dogs


# Dataset Format
   PetImages/
├── Cat/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── Dog/
    ├── 1.jpg
    ├── 2.jpg
    └── ...

   # Output Example
   Loading data...
Data Loaded Successfully!
Total images loaded: 300
Splitting data...
Training model...
Evaluating model...

Model Performance:
              precision    recall  f1-score   support
           0       0.85      0.81      0.83        30
           1       0.80      0.84      0.82        30
    accuracy                           0.82        60

Execution Completed!
Summary:
- Total Images Used  : 300
- Train/Test Split   : 240/60
- Model Used         : SVM (Support Vector Machine)
- Accuracy Achieved  : 0.82


#Future Improvements
Use CNN features (e.g., MobileNetV2 or VGG16 embeddings)
Add GUI or web interface
Hyperparameter tuning
Real-time image prediction
Confusion matrix visualization

Author
Sanjana 


