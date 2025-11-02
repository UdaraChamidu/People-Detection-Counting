from ultralytics import YOLO
import cv2
import numpy as np
from tracker import Tracker

# Load COCO classes
with open('coco.txt', 'r') as f:
    class_list = f.read().split('\n')

# Initialize model and tracker
model = YOLO('yolov8s.pt')
tracker = Tracker()

# Define ROI areas (you'll fill these with your coordinates)
area1 = np.array([[1366, 210], [1527, 314], [1527, 446], [1223, 212]])     
area2 = np.array([[1088, 515], [1142, 424], [1522, 592], [1454, 739]])     

# Tracking sets
entering = set()
exiting = set()

# Dictionary to store people positions
people_entering = {}
people_exiting = {}

# Mouse callback for getting coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"X: {x}, Y: {y}")

# Load video
cap = cv2.VideoCapture('video.mp4')

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Process detections
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = class_list[cls]
            
            if 'person' in class_name:
                detections.append([x1, y1, x2, y2])
    
    # Update tracker
    bbox_id = tracker.update(detections)
    
    # Draw ROI zones
    cv2.polylines(frame, [area1], True, (0, 255, 0), 2)
    cv2.polylines(frame, [area2], True, (0, 255, 0), 2)
    cv2.putText(frame, '1', (550, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(frame, '2', (1150, 400), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    # Process tracked objects
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        
        # Draw rectangle and ID for ALL tracked people
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)  # Purple rectangle for all people
        cv2.circle(frame, (x4, y4), 4, (0, 255, 255), -1)  # Yellow circle at bottom-right
        cv2.putText(frame, str(id), (x3, y3-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        
        # Check if in Area 2 (entry detection)
        result2 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if result2 >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 3)  # Yellow when in Area 2
        
        # Check if in Area 1 (and was in Area 2) - ENTERING
        if id in people_entering:
            result1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if result1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)  # Green when entering
                entering.add(id)
                
        # Check if in Area 1 (exit detection)
        result1_exit = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if result1_exit >= 0:
            people_exiting[id] = (x4, y4)
            
        # Check if in Area 2 (and was in Area 1) - EXITING
        if id in people_exiting:
            result2_exit = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if result2_exit >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)  # Red when exiting
                exiting.add(id)
    
    # Display counters with background for better visibility
    cv2.rectangle(frame, (40, 20), (350, 120), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, f'Entering: {len(entering)}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exiting: {len(exiting)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Changed from 1 to 30 for smoother playback
        break

cap.release()
cv2.destroyAllWindows()

