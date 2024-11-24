from ultralytics import YOLO
import cv2

# Load custom YOLOv8 model
model_path = 'best10s.pt'
model = YOLO(model_path)

# Dictionary of class names corresponding to the model's classes
class_names = {
    0: 'Bien bao ben xe bus',
    1: 'Bien bao cam cac xe o to tai co khoi luong chuyen cho',
    2: 'Bien bao cam di nguoc chieu',
    3: 'Bien bao cam di thang',
    4: 'Bien bao cam do xe',
    5: 'Bien bao cam dung xe va do xe',
    6: 'Bien bao cam nguoi di bo',
    7: 'Bien bao cam oto re trai va quay dau xe',
    8: 'Bien bao cam quay dau xe',
    9: 'Bien bao cam re phai',
    10: 'Bien bao cam re phai va quay dau',
    11: 'Bien bao cam re trai',
    12: 'Bien bao cam re trai va quay dau',
    13: 'Bien bao cam vuot',
    14: 'Bien bao cam xe may',
    15: 'Bien bao cam xe may re trai',
    16: 'Bien bao cam xe o to re phai',
    17: 'Bien bao cam xe o to re trai',
    18: 'Bien bao cam xe oto',
    19: 'Bien bao cam xe oto khach va xe oto tai',
    20: 'Bien bao cam xe oto tai',
    21: 'Bien bao cam xe so-mi ro-mooc',
    22: 'Bien bao chieu cao tinh khong thuc te',
    23: 'Bien bao cho ngoat nguy hiem',
    24: 'Bien bao cho ngoat nguy hiem vong ben trai',
    25: 'Bien bao cho quay xe',
    26: 'Bien bao cong truong',
    27: 'Bien bao di cham',
    28: 'Bien bao doc xuong nguy hiem',
    29: 'Bien bao duong bi thu hep',
    30: 'Bien bao duong cao toc phia truoc',
    31: 'Bien bao duong co go giam toc',
    32: 'Bien bao duong giao nhau',
    33: 'Bien bao duong nguoi di bo cat ngang',
    34: 'Bien bao giao nhau co tin hieu den',
    35: 'Bien bao giao nhau voi duong khong uu tien',
    36: 'Bien bao giao nhau voi duong uu tien',
    37: 'Bien bao han che chieu cao',
    38: 'Bien bao huong di phai theo',
    39: 'Bien bao huong di thang phai theo',
    40: 'Bien bao huong phai di vong chuong ngai vat',
    41: 'Bien bao lan duong danh cho o to con',
    42: 'Bien bao lan duong danh cho xe con va xe buyt',
    43: 'Bien bao lan duong danh cho xe may va xe 3 banh',
    44: 'Bien bao nguy hiem khac',
    45: 'Bien bao nhieu cho ngoat nguy hiem lien tiep',
    46: 'Bien bao noi giao nhau chay theo vong xuyen',
    47: 'Bien bao toc do toi da cho phep',
    48: 'Bien bao toc do toi thieu cho phep',
    49: 'Bien bao tre em',
    50: 'Bien hieu lenh bat dau khu dong dan cu',
    51: 'Bien hieu lenh het khu dong dan cu'
}


# Function to draw bounding boxes and class labels on the frame
def plot_boxes(results, frame, class_names):
    for box in results[0].boxes:  # Iterate through detected boxes
        # Extract bounding box coordinates (xyxy), confidence, and class label
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf.item()
        label = int(box.cls.item())

        # Get the class name from the label
        class_name = class_names.get(label, 'Unknown')

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Capture video from a file or webcam (change 0 to a file path if you want to use a video file)
video_path = 'test_6.mp4'  # Use '0' for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()  # Read frame-by-frame
    if not ret:
        break  # Exit the loop if no frame is returned

    # Perform object detection on the current frame
    results = model.predict(frame)

    # Draw bounding boxes and labels
    frame = plot_boxes(results, frame, class_names)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Press 'q' to quit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
