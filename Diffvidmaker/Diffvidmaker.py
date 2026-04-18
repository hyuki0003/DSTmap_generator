import cv2
import os
import numpy as np


def compute_frame_difference(input_video_path):
    if not os.path.exists(input_video_path):
        print(f"Error: 파일 {input_video_path}이(가) 존재하지 않습니다.")
        return []

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: 비디오를 열 수 없습니다.")
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: 첫 번째 프레임을 읽을 수 없습니다.")
        return []

    frame_diffs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # RGB 채널별로 프레임 차이 계산
        b_diff = frame[:, :, 0].astype(np.int16) - prev_frame[:, :, 0].astype(np.int16)
        g_diff = frame[:, :, 1].astype(np.int16) - prev_frame[:, :, 1].astype(np.int16)
        r_diff = frame[:, :, 2].astype(np.int16) - prev_frame[:, :, 2].astype(np.int16)

        frame_diff = cv2.merge((b_diff, g_diff, r_diff))
        frame_diffs.append(frame_diff)

        prev_frame = frame

    cap.release()
    return frame_diffs


def save_frame_differences(frame_diffs, output_video_path, fps, frame_size):
    # 비디오 라이터 객체 생성 (avi 파일을 위한 MJPG 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=False)

    for diff in frame_diffs:
        out.write(diff)

    # 자원 해제
    out.release()
