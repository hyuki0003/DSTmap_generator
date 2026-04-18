import cv2
import os
import numpy as np


def compute_frame_difference(input_video_path):
    if not os.path.exists(input_video_path):
        print(f"Error: File {input_video_path} does not exist.")
        return []

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # 첫 프레임 읽기
    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame.")
        return []

    frame_diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이전 프레임과 현재 프레임 간 채널별 차이 계산
        b_diff = cv2.absdiff(prev_frame[:, :, 0], frame[:, :, 0])
        g_diff = cv2.absdiff(prev_frame[:, :, 1], frame[:, :, 1])
        r_diff = cv2.absdiff(prev_frame[:, :, 2], frame[:, :, 2])

        # 채널별 차이를 합산하여 차이 영상을 생성
        frame_diff = cv2.merge((b_diff, g_diff, r_diff))

        # 차이 영상을 리스트에 추가
        frame_diffs.append(frame_diff)

        # 현재 프레임을 이전 프레임으로 설정
        prev_frame = frame

    # 자원 해제
    cap.release()

    return frame_diffs


def normalize_frames(frame_diffs):
    normalized_diffs = []
    for diff in frame_diffs:
        # 각 채널별로 0-255 범위로 노말라이즈
        b, g, r = cv2.split(diff)
        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        normalized_diff = cv2.merge((b, g, r))
        normalized_diffs.append(normalized_diff.astype(np.uint8))
    return normalized_diffs


def save_frame_differences(frame_diffs, output_video_path, fps, frame_size):
    # 비디오 라이터 객체 생성 (avi 파일을 위한 MJPG 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=True)

    for diff in frame_diffs:
        out.write(diff)

    # 자원 해제
    out.release()


# 사용 예시
input_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/vid.avi'
output_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/DIFFVideos_v2/v2_BN_output_video.avi'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
else:
    # 비디오의 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # 자원 해제
    cap.release()

    # 프레임 간 차이 계산
    frame_diffs = compute_frame_difference(input_video_path)

    # 프레임 차이 정규화
    normalized_diffs = normalize_frames(frame_diffs)

    # 차이 영상을 비디오 파일로 저장
    if normalized_diffs:
        save_frame_differences(normalized_diffs, output_video_path, fps, frame_size)

