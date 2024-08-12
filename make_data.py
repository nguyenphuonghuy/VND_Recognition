import numpy as np
import cv2
import time
import os

# Label: 0k là ko cầm tiền, còn lại là các mệnh giá
label = "000000"

cap = cv2.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
for i in range(1062):

    # Capture frame-by-frame

    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None,fx=0.3,fy=0.3)

    # Hiển thị
    cv2.imshow('frame',frame)

    # Lưu dữ liệu
    if i>60:
        print("Số ảnh capture = ",i-60)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png",frame)
    # nhan q ngat cap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()