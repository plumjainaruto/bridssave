import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")  # ระบุ path โมเดลของคุณ

# แสดงคลาสที่มีในโมเดล
print("Classes in the model:", model.names)  # แสดงรายชื่อคลาสทั้งหมดในโมเดล

# เปิดกล้อง
cap = cv2.VideoCapture(1)  # ใส่ 0 สำหรับกล้องในเครื่อง หรือเปลี่ยนเป็น URL/IP สำหรับกล้อง IP

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านข้อมูลจากกล้องได้!")
        break

    # ใช้โมเดล YOLOv8 ในการตรวจจับ
    results = model.predict(source=frame, conf=0.25)  # conf=0.25 คือค่า confidence threshold
    annotated_frame = results[0].plot()  # วาดกรอบผลลัพธ์บนภาพ

    # แสดงผลในหน้าต่าง
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
