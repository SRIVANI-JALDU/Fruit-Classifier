import cv2
import numpy as np
import joblib
from skimage.feature import hog
from imutils.video import VideoStream
import time
import os

# Load model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fruit_quality_svm_model.pkl')

def load_model():
    """Load the trained SVM model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first using predefined.py")
        exit()
    
    model_data = joblib.load(MODEL_PATH)
    return (model_data['model'], 
            model_data['pca'], 
            model_data['scaler'],
            model_data['img_size'])

def extract_features(image, img_size):
    """Feature extraction matching the training pipeline"""
    # Check if image is valid
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for feature extraction")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    resized = cv2.resize(image, img_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # HOG features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
    
    # Color histograms
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    # Normalize and concatenate
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    return np.hstack([hog_features, hist_h, hist_s, hist_v])

def main():
    # Load model
    svm_model, pca, scaler, img_size = load_model()
    labels = ['Fresh', 'Rotten']
    colors = [(0, 255, 0), (0, 0, 255)]  # Green, Red
    
    # Initialize video stream
    print("Starting video stream...")
    print("If camera doesn't open, try changing src=0 to src=1 in VideoStream()")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    roi = None
    roi_selected = False
    
    while True:
        frame = vs.read()
        if frame is None:
            print("Error: Could not read frame from camera")
            print("Possible fixes:")
            print("1. Check camera connection")
            print("2. Try changing VideoStream(src=0) to src=1")
            break
            
        display_frame = frame.copy()
        
        # Instructions
        cv2.putText(display_frame, "Press 's' to select ROI", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'c' to classify", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show ROI selection
        if roi_selected:
            x, y, w, h = roi
            # Validate ROI coordinates
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x+w <= frame.shape[1] and y+h <= frame.shape[0]):
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                print("Warning: Invalid ROI - please select again")
                roi_selected = False
        
        key = cv2.waitKey(1) & 0xFF
        
        # Select ROI
        if key == ord("s"):
            print("\n=== ROI Selection Instructions ===")
            print("1. Click and drag to select fruit region")
            print("2. Press SPACE or ENTER to confirm")
            print("3. Press C to cancel selection\n")
            
            roi = cv2.selectROI("Select Fruit Region (Confirm=ENTER, Cancel=C)", 
                              frame, 
                              showCrosshair=True)
            cv2.destroyWindow("Select Fruit Region (Confirm=ENTER, Cancel=C)")
            
            # Check if valid selection was made
            if roi[2] > 0 and roi[3] > 0:  # Check width and height
                roi_selected = True
                print(f"ROI successfully selected at coordinates: {roi}")
            else:
                roi_selected = False
                print("ROI selection cancelled or invalid")
        
        # Classify
        elif key == ord("c"):
            if not roi_selected:
                print("Error: Please select ROI first by pressing 's'")
                continue
                
            x, y, w, h = roi
            # Validate ROI before processing
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x+w <= frame.shape[1] and y+h <= frame.shape[0]):
                fruit_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Extract features and predict
                    features = extract_features(fruit_roi, img_size)
                    features_scaled = scaler.transform([features])
                    features_pca = pca.transform(features_scaled)
                    
                    prediction = svm_model.predict(features_pca)[0]
                    proba = svm_model.predict_proba(features_pca)[0]
                    
                    # Display results
                    label = f"{labels[prediction]} ({proba[prediction]:.2f})"
                    color = colors[prediction]
                    
                    cv2.putText(display_frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    print(f"Classification result: {label}")
                except Exception as e:
                    print(f"Error during classification: {str(e)}")
            else:
                print("Error: Invalid ROI - please select a new region")
                roi_selected = False
    
        elif key == ord("q"):
            break
        
        cv2.imshow("Fruit Quality Classifier", display_frame)
   
    cv2.destroyAllWindows()
    vs.stop()
    print("Program terminated successfully")

if __name__ == "__main__":
    main()