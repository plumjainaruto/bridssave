import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")  # ระบุ path โมเดลของคุณ

# โหลดภาพจากไฟล์
image_path = "320.png"  # ใช้ไฟล์ที่ถูกต้องในโฟลเดอร์เดียวกัน
image = cv2.imread(image_path)  # อ่านไฟล์ภาพ

if image is None:
    print("ไม่สามารถโหลดภาพได้! ตรวจสอบ path หรือชื่อไฟล์")
    exit()

# ใช้โมเดล YOLOv8 ในการตรวจจับ
results = model.predict(source=image, conf=0.25)  # conf=0.25 คือค่า confidence threshold
annotated_image = results[0].plot()  # วาดกรอบผลลัพธ์บนภาพ

# แสดงผลลัพธ์ในหน้าต่าง
cv2.imshow("YOLOv8 Object Detection", annotated_image)

# กดปุ่มใดๆ เพื่อปิดหน้าต่าง
cv2.waitKey(0)
cv2.destroyAllWindows()

# (ไม่บังคับ) บันทึกภาพผลลัพธ์ที่ตรวจจับลงไฟล์
cv2.imwrite("output.jpg", annotated_image)
