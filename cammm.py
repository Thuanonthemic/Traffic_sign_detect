import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
model=YOLO('yolov8n.pt')
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
    24: 'Bien bao cho quay xe',
    25: 'Bien bao cong truong',
    26: 'Bien bao di cham',
    27: 'Bien bao doc xuong nguy hiem',
    28: 'Bien bao duong bi thu hep',
    29: 'Bien bao duong cao toc phia truoc',
    30: 'Bien bao duong co go giam toc',
    31: 'Bien bao duong giao nhau',
    32: 'Bien bao duong nguoi di bo cat ngang',
    33: 'Bien bao giao nhau co tin hieu den',
    34: 'Bien bao giao nhau voi duong khong uu tien',
    35: 'Bien bao giao nhau voi duong uu tien',
    36: 'Bien bao han che chieu cao',
    37: 'Bien bao huong di phai theo',
    38: 'Bien bao huong di thang phai theo',
    39: 'Bien bao huong phai di vong chuong ngai vat',
    40: 'Bien bao lan duong danh cho o to con',
    41: 'Bien bao lan duong danh cho xe con va xe buyt',
    42: 'Bien bao lan duong danh cho xe may va xe 3 banh',
    43: 'Bien bao nguy hiem khac',
    44: 'Bien bao nhieu cho ngoat nguy hiem lien tiep',
    45: 'Bien bao noi giao nhau chay theo vong xuyen',
    46: 'Bien bao toc do toi da cho phep',
    47: 'Bien bao toc do toi thieu cho phep',
    48: 'Bien bao tre em',
    49: 'Bien hieu lenh bat dau khu dong dan cu',
    50: 'Bien hieu lenh het khu dong dan cu'
}

fps_start_time = time.time()
frame_count = 0

while True:
    im= picam2.capture_array()

    frame_count += 1

    if time.time() - fps_start_time >= 1:
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    im=cv2.flip(im,-1)
    results=model.predict(im)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        # Use the dictionary to get the corresponding class name
        c = class_names[d]

        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)

 cvzone.putTextRect(im, f'FPS: {fps:.2f}', (10, 30), 1, 2)

    cv2.imshow("Camera", im)
    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()