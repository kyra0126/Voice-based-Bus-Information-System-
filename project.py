import cv2
import pytesseract
from gtts import gTTS
import os
import numpy as np
import time
# pytesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def detect_motion(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def post_process(thresh):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated = cv2.dilate(opening, kernel, iterations=2)
    return dilated

def detect_bus(frame1, frame2):
    pre_frame1 = preprocess_frame(frame1)
    pre_frame2 = preprocess_frame(frame2)
    motion = detect_motion(pre_frame1, pre_frame2)
    processed = post_process(motion)
    
    # 노란색 번호판 감지
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    hsv_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    
    # 형태학적 연산
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated_yellow_areas = cv2.dilate(opening, kernel, iterations=2)
    
    # 움직임이 있는 노란색 영역과 processed 결과의 교집합
    combined = cv2.bitwise_and(dilated_yellow_areas, processed)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 큰 컨투어 분석
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 4000:  # 번호판 크기에 맞게 임계값 조정
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 3)
            return True, frame2, combined, (x, y, w, h)
    return False, frame2, combined, None

def extract_bus_number(frame, bbox):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    cv2.imwrite('blur_roi.jpg', blur_roi)

    text = pytesseract.image_to_string(blur_roi, lang='kor', config='--psm 8 --oem 0')
    return text

def is_approaching(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    return mean_mag > 1.0  # 임계값 조정 필요

def main():
    start = time.time()
    cap = cv2.VideoCapture('bus_stop_video_12fps.mp4')
    ret, prev_frame = cap.read()
    if not ret:
        print("비디오를 읽을 수 없습니다.")
        return

    # height, width = prev_frame.shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('bus_detected_video.mp4', fourcc, 20.0, (width, height))
    # out_dilated = cv2.VideoWriter('dilated_video.mp4', fourcc, 20.0, (width, height), isColor=False)

    announced_buses = {}  # 버스 번호와 접근 상태를 저장할 딕셔너리

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bus_detected, processed_frame, dilated_frame, bbox = detect_bus(prev_frame, frame)
        if bus_detected and bbox is not None:
            bus_number = extract_bus_number(frame, bbox)
            approaching = is_approaching(prev_frame, frame)
            direction = "접근 중" if approaching else "멀어지는 중"

            if bus_number not in announced_buses or announced_buses[bus_number] != direction:
                # tts = gTTS(text=f'버스 번호 {bus_number}번 {direction}', lang='ko')
                print(f'버스 번호 {bus_number}번 {direction}')
                # tts.save('bus_info.mp3')
                # os.system('open bus_info.mp3')
                announced_buses[bus_number] = direction

        prev_frame = frame
        # out.write(processed_frame)
        # out_dilated.write(dilated_frame)

    cap.release()
    # out.release()
    # out_dilated.release()
    end = time.time()
    print(f"총 소요 시간: {end - start:.2f}초")

if __name__ == "__main__":
    main()