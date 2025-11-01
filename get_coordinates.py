import cv2

# Global list to store coordinates
coordinates = []

def mouse_callback(event, x, y, flags, param):
    global coordinates
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click to add coordinate
        coordinates.append([x, y])
        print(f"Coordinate {len(coordinates)}: X={x}, Y={y}")
        
        # Draw a circle at clicked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Get Coordinates', frame)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        # Show current mouse position
        temp_frame = frame.copy()
        cv2.putText(temp_frame, f'X: {x}, Y: {y}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Get Coordinates', temp_frame)

# Load your video
cap = cv2.VideoCapture('video.mp4')  # Replace with your video filename

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video file!")
    exit()

# Create window and set mouse callback
cv2.namedWindow('Get Coordinates')
cv2.setMouseCallback('Get Coordinates', mouse_callback)

print("\n=== INSTRUCTIONS ===")
print("1. Click 4 points for Area 1 (in order: top-left, top-right, bottom-right, bottom-left)")
print("2. Click 4 points for Area 2 (same order)")
print("3. Press 'r' to reset coordinates")
print("4. Press 'q' when done")
print("====================\n")

cv2.imshow('Get Coordinates', frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset coordinates
        coordinates = []
        ret, frame = cap.read()
        cv2.imshow('Get Coordinates', frame)
        print("Coordinates reset!")

cap.release()
cv2.destroyAllWindows()

# Print the final coordinates
print("\n=== YOUR COORDINATES ===")
if len(coordinates) >= 4:
    print(f"\narea1 = np.array([{coordinates[0]}, {coordinates[1]}, {coordinates[2]}, {coordinates[3]}])")
if len(coordinates) >= 8:
    print(f"area2 = np.array([{coordinates[4]}, {coordinates[5]}, {coordinates[6]}, {coordinates[7]}])")
print("\nCopy these lines into your main.py file!")
