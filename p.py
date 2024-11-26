import cv2
from ultralytics import YOLO
import pygame

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")  # ระบุ path โมเดลของคุณ

# ตั้งค่าเล่นเสียงด้วย pygame
pygame.mixer.init()  # เริ่มต้นระบบเสียง
sound_pigeon_bird = pygame.mixer.Sound("20000.mp3")  # โหลดไฟล์เสียงสำหรับ Pigeon และ bird
sound_sparrow = pygame.mixer.Sound("100.mp3")  # โหลดไฟล์เสียงสำหรับ sparrow

# เปิดกล้อง
cap = cv2.VideoCapture(0)  # ใส่ 0 สำหรับกล้องในเครื่อง หรือเปลี่ยนเป็น URL/IP สำหรับกล้อง IP

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

    # ตรวจสอบว่ามีคลาสที่ต้องการถูกตรวจพบหรือไม่
    for result in results[0].boxes.data:  # วนลูปเช็คกล่องที่ตรวจจับได้
        class_id = int(result[5])  # class_id คือ index ของคลาสที่ตรวจพบ
        if class_id == 0 or class_id == 1:  # ตรวจจับ Pigeon หรือ bird
            print(f"พบคลาส: {model.names[class_id]}")  # แสดงชื่อคลาสที่พบ
            sound_pigeon_bird.play()  # เล่นเสียง 20000.mp3
        elif class_id == 2:  # ตรวจจับ sparrow
            print(f"พบคลาส: {model.names[class_id]}")  # แสดงชื่อคลาสที่พบ
            sound_sparrow.play()  # เล่นเสียง 100.mp3

    # แสดงผลในหน้าต่าง
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # ปิดระบบเสียง
