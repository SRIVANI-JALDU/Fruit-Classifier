
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'fruit_quality_dataset')
categories = ['fresh', 'rotten']
img_size = (128, 128)

def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    resized = cv2.resize(image, img_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.hstack([hog_features, hist_h, hist_s, hist_v])
def load_dataset():
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(DATASET_PATH, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                features = extract_features(img)
                data.append(features)
                labels.append(class_num)
    
    return np.array(data), np.array(labels)
def main():
    print("Loading dataset...")
    X, y = load_dataset()
    
    print("\nDataset Summary:")
    print(f"Total images: {len(X)}")
    print(f"Fresh fruits: {np.sum(y == 0)}")
    print(f"Rotten fruits: {np.sum(y == 1)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print("\nTraining SVM model...")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', 
                   class_weight='balanced', probability=True)
    svm_model.fit(X_train_pca, y_train)
    
    y_pred = svm_model.predict(X_test_pca)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=categories))

    joblib.dump({
        'model': svm_model,
        'pca': pca,
        'scaler': scaler,
        'img_size': img_size
    }, 'fruit_quality_svm_model.pkl')
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()
