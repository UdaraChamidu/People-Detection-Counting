from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('yolov8s.pt')

# Test with webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Show results
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Test', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
