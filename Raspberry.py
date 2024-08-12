import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import RPi.GPIO as GPIO
import time
import speech_recognition as sr
from gtts import gTTS
import pygame
import os

# Đường dẫn tới mô hình đã lưu
model_path = '/home/duy/code/money_recognition/student1_model.h5'

# Từ điển chứa các lớp phân loại
categories = {
    0: "000000",
    1: "000200",
    2: "000500",
    3: "001000",
    4: "002000",
    5: "005000",
    6: "010000",
    7: "020000",
    8: "050000",
    9: "100000",
    10: "200000",
    11: "500000",
}

# GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)

# Set GPIO Pins for ultrasonic sensor
GPIO_TRIGGER = 18
GPIO_ECHO = 24

# Set GPIO direction (IN / OUT) for ultrasonic sensor
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Initialize pygame mixer
pygame.mixer.init()

# Function to speak Vietnamese text using gTTS
def speak_vietnamese(text):
    tts = gTTS(text, lang='vi')
    tts.save("temp.mp3")
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    os.remove("temp.mp3")

# Function to measure distance using ultrasonic sensor
def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    dist = (TimeElapsed * 34300) / 2

    return dist

# Tiền xử lý ảnh
def preprocess_image(img, target_size):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') 
    img = np.expand_dims(img, axis=0)
    return img

# Tải mô hình
model = load_model(model_path)

# Function to open the camera
def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        speak_vietnamese("Không thể mở camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            speak_vietnamese("Không nhận được khung hình. Đang thoát...")
            break

        # Tiền xử lý khung hình
        img_array = preprocess_image(frame, (128, 128))

        # Dự đoán lớp của ảnh
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Hiển thị kết quả dự đoán lên khung hình
        label = f"Predicted: {categories[predicted_class]}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Camera', frame)

        # Phát âm thanh loại tiền nếu độ tin cậy trên 80%
        confidence = prediction[0][predicted_class] * 100
        if confidence > 80:
            speak_vietnamese(f"Đây là tiền {categories[predicted_class]}")

        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to recognize speech
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Đang lắng nghe...")
        speak_vietnamese("Đang lắng nghe...")
        audio_data = r.listen(source)
        try:
            # Recognize speech using Google's speech recognition
            command = r.recognize_google(audio_data, language="vi-VN")
            print("Bạn nói:", command)
            speak_vietnamese("Bạn nói: " + command)
            return command.lower()
        except sr.UnknownValueError:
            print("Google không nhận diện được âm thanh")
            speak_vietnamese("Google không nhận diện được âm thanh")
        except sr.RequestError as e:
            print("Không thể yêu cầu kết quả từ dịch vụ Google; {0}".format(e))
            speak_vietnamese("Không thể yêu cầu kết quả từ dịch vụ Google")

# Flag to track the state of distance sensing
distance_sensing_enabled = True

if __name__ == "__main__":
    try:
        while True:
            # Đo khoảng cách nếu chức năng được kích hoạt
            if distance_sensing_enabled:
                dist = distance()
                if dist < 100:
                    text_to_speak = "Khoảng cách đo được là %.1f centimet" % dist
                    speak_vietnamese(text_to_speak)
                    print("Measured Distance = %.1f cm" % dist)

            # Lấy lệnh từ âm thanh
            command = recognize_speech()
            if command:
                if "mở camera" in command:
                    print("Đang mở camera...")
                    speak_vietnamese("Đang mở camera...")
                    # Tạm dừng chức năng đo khoảng cách
                    distance_sensing_enabled = False
                    open_camera()
                    # Khởi động lại chức năng đo khoảng cách sau khi đóng camera
                    distance_sensing_enabled = True
                elif "đóng camera" in command or "thoát" in command:
                    print("Đang đóng camera...")
                    speak_vietnamese("Đang đóng camera...")
                    break

    except KeyboardInterrupt:
        print("Measurement stopped by User")
    finally:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows
