import os
from ultralytics import YOLO
import cv2

# Open the default camera (0 for the first camera, 1 for the second, etc.)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Get the frame dimensions
ret, frame = cap.read()
H, W, _ = frame.shape

# Create a VideoWriter object for the output video file
output_path = 'camera_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'best.pt')
# Load the YOLOv8 model
model = YOLO(model_path)
threshold = 0.5

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to read a frame from the camera.")
        break

    # Perform object detection
    results = model(frame)[0]
    
    # Dictionary to keep count of objects detected
    object_counter = {}

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            class_name = results.names[int(class_id)].upper()
            
            # Update the object count
            if class_name in object_counter:
                object_counter[class_name] += 1
            else:
                object_counter[class_name] = 1
            
            # Add numbering to the label
            label = f"{class_name}{object_counter[class_name]} {score:.2f}"
            
            # Draw the bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the processed frame (optional)
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()   
out.release()
cv2.destroyAllWindows()
