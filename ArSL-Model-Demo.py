import cv2 
import numpy as np 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array 
 
# Load the trained model 
model = load_model('arabic_sign_language_model.h5') 
 
# Define the labels 
labels = ['Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh', 'Jeem', 
          'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad', 'Seen', 'Sheen', 
          'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'] 
 
# Initialize the webcam 
cap = cv2.VideoCapture(0) 
 
# Check if the webcam opened successfully 
if not cap.isOpened(): 
    print("Error: Could not open webcam.") 
    exit() 
 
def preprocess_frame(frame): 
    # Resize the frame to the input size expected by the model 
    resized_frame = cv2.resize(frame, (128, 128)) 
    img_array = img_to_array(resized_frame) / 255.0  # Normalize pixel values 
    return np.expand_dims(img_array, axis=0) 
 
while True: 
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    # Flip the frame horizontally for natural interaction 
    frame = cv2.flip(frame, 1) 
 
    # Preprocess the frame for prediction 
    processed_frame = preprocess_frame(frame) 
 
    # Make prediction 
    predictions = model.predict(processed_frame) 
    class_index = np.argmax(predictions) 
    class_label = labels[class_index] 
 
    # Display the label on the frame 
    cv2.putText(frame, f"Predicted Sign: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
 
    # Show the frame 
    cv2.imshow('Arabic Sign Language Recognition', frame) 
 
    # Break the loop if 'q' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
 
# Release the capture and close windows 
cap.release() 
cv2.destroyAllWindows()
