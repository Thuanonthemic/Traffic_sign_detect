from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

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

# Load the image
image_path = 'hinh_12.png'
image = cv2.imread(image_path)

# Perform object detection
results = model.predict(image)

# Display the results
def plot_boxes(results, image, class_names):
    for box in results[0].boxes:  # Iterate through detected boxes
        # Extract bounding box coordinates (xyxy), confidence, and class label
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        confidence = box.conf.item()  # Confidence score
        label = int(box.cls.item())   # Class label (integer)

        # Get the class name from the label
        class_name = class_names.get(label, 'Unknown')  # Use .get() to handle unknown labels

        # Draw the bounding box on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} {confidence:.2f}', (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with boxes using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Plot the detection results with class names
plot_boxes(results, image, class_names)

