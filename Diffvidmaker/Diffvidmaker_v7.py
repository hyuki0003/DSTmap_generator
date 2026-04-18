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


def move_variation(frame_diffs):
    moved_frames = []

    for diff in frame_diffs:
        b, g, r = cv2.split(diff)

        # 각 채널을 -255에서 255 범위에서 0에서 1 범위로 스케일링
        b = (b + 255.0) / (2 * 255.0)
        g = (g + 255.0) / (2 * 255.0)
        r = (r + 255.0) / (2 * 255.0)

        moved = cv2.merge((b, g, r))
        moved_frames.append(moved)

    return moved_frames


def find_global_min_max(frame_diffs):
    min_val = np.array([np.inf, np.inf, np.inf])
    max_val = np.array([-np.inf, -np.inf, -np.inf])

    for diff in frame_diffs:
        b, g, r = cv2.split(diff)

        min_val[0] = min(min_val[0], np.min(b))
        min_val[1] = min(min_val[1], np.min(g))
        min_val[2] = min(min_val[2], np.min(r))

        max_val[0] = max(max_val[0], np.max(b))
        max_val[1] = max(max_val[1], np.max(g))
        max_val[2] = max(max_val[2], np.max(r))

    # 디버그 출력
    # print("find_global_min_max - min_val:", min_val)
    # print("find_global_min_max - max_val:", max_val)

    return min_val, max_val


def normalize_frames(frame_diffs, global_min, global_max):
    normalized_diffs = []

    for diff in frame_diffs:
        b, g, r = cv2.split(diff)

        # 각 채널을 전체 최소값과 최대값을 사용하여 0에서 255 범위로 정규화
        b = 255.0 * (b - global_min[0]) / (global_max[0] - global_min[0] + 1e-5)
        g = 255.0 * (g - global_min[1]) / (global_max[1] - global_min[1] + 1e-5)
        r = 255.0 * (r - global_min[2]) / (global_max[2] - global_min[2] + 1e-5)

        # 디버그 출력
        # print("normalize_frames - b stats:", np.min(b), np.max(b), np.isnan(b).sum())
        # print("normalize_frames - g stats:", np.min(g), np.max(g), np.isnan(g).sum())
        # print("normalize_frames - r stats:", np.min(r), np.max(r), np.isnan(r).sum())

        b = np.clip(b, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        r = np.clip(r, 0, 255).astype(np.uint8)

        normalized_diff = cv2.merge((b, g, r))
        normalized_diffs.append(normalized_diff)

    return normalized_diffs

def remove_high_intensity_pixels(frame_diffs, threshold=135):

    processed_diffs = []
    for diff in frame_diffs:

        mask = np.any(diff >= threshold, axis=-1)
        processed_diff = diff.copy()

        processed_diff[mask] = [0, 0, 0]

        processed_diffs.append(processed_diff)

    return processed_diffs

def save_frame_differences(frame_diffs, output_video_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=True)

    if not out.isOpened():
        print(f"Error: 출력 비디오 파일 {output_video_path}을(를) 열 수 없습니다.")
        return

    for diff in frame_diffs:
        out.write(diff)

    out.release()
    print(f"비디오가 성공적으로 {output_video_path}에 저장되었습니다.")


def normalize_and_process_frames(frame_diffs):
    moved_diffs = move_variation(frame_diffs)
    min_val, max_val = find_global_min_max(moved_diffs)
    processed_diffs = normalize_frames(moved_diffs, min_val, max_val)
    processed_diffs = remove_high_intensity_pixels(processed_diffs)
    return processed_diffs


if __name__ == "__main__":
    input_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/vid.avi'
    output_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/DIFFVideos_v6/v6-1_output_video.avi'

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: 비디오 파일 {input_video_path}을(를) 열 수 없습니다.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)

        cap.release()

        frame_diffs = compute_frame_difference(input_video_path)

        if frame_diffs:
            processed_diffs = normalize_and_process_frames(frame_diffs)
            save_frame_differences(processed_diffs, output_video_path, fps, frame_size)
