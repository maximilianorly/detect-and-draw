import sys
sys.path.append('yolov5')
import cv2
import torch
from PIL import Image
import numpy as np

# Import YOLOv5 code from the cloned repo
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Load YOLOv5 model
weights_path = "weights/yolov5s.pt"
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path)

# Load the input image
input_image_path = "image.jpg"
img = cv2.imread(input_image_path)

# Resize the image to match the expected model input size (640x640 for yolov5s)
input_size = (640, 640)
img = cv2.resize(img, input_size)

# Convert the image to the correct format (3 channels) and data type (float32)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

# Normalize the image to [0, 1] range
img /= 255.0

# Perform object detection
# Convert the NumPy array to a PyTorch tensor
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

# Perform object detection
results = model(img_tensor)

# Specify the class name of the object you want to detect (e.g., "person")
target_class_name = "car"

# Get detection results from the model output
predictions = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)
print(predictions)

# Find the specified object class
for pred in predictions[0]:
    class_index = int(pred[5])
    class_score = float(pred[4])
    class_names = model.names

    # Continue with medium/high confidence
    if class_names[class_index] == target_class_name and class_score > 0.5:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, pred[:4])

        # Draw bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the class name and confidence
        label = f"{class_names[class_index]}: {class_score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the annotated image
output_image_path = "output_image.jpg"
cv2.imwrite(output_image_path, img)

# Display the image with bounding boxes
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()