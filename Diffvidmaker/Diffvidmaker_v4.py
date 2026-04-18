import cv2
import os
import numpy as np

def compute_frame_difference(input_video_path):
    if not os.path.exists(input_video_path):
        print(f"Error: File {input_video_path} does not exist.")
        return []

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame.")
        return []

    frame_diffs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        b_diff = cv2.absdiff(prev_frame[:, :, 0], frame[:, :, 0])
        g_diff = cv2.absdiff(prev_frame[:, :, 1], frame[:, :, 1])
        r_diff = cv2.absdiff(prev_frame[:, :, 2], frame[:, :, 2])

        frame_diff = cv2.merge((b_diff, g_diff, r_diff))

        frame_diffs.append(frame_diff)

        prev_frame = frame

    cap.release()

    return frame_diffs

def find_min_max(frame_diffs):
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

    return min_val, max_val

def normalize_frames(frame_diffs, min_val, max_val):
    normalized_diffs = []
    for diff in frame_diffs:
        b, g, r = cv2.split(diff)

        b_range = max_val[0] - min_val[0] if max_val[0] != min_val[0] else 1
        g_range = max_val[1] - min_val[1] if max_val[1] != min_val[1] else 1
        r_range = max_val[2] - min_val[2] if max_val[2] != min_val[2] else 1

        b = (b - min_val[0]) * 255.0 / b_range
        g = (g - min_val[1]) * 255.0 / g_range
        r = (r - min_val[2]) * 255.0 / r_range

        b = np.clip(b, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        r = np.clip(r, 0, 255).astype(np.uint8)

        normalized_diff = cv2.merge((b, g, r))
        normalized_diffs.append(normalized_diff)

    return normalized_diffs

def remove_high_intensity_pixels(frame_diffs, threshold=50):
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
        print(f"Error: Could not open the output video file {output_video_path}")
        return

    for diff in frame_diffs:
        out.write(diff)

    out.release()
    print(f"Video saved successfully to {output_video_path}")

def normalize_and_process_frames(frame_diffs):
    min_val, max_val = find_min_max(frame_diffs)
    normalized_diffs = normalize_frames(frame_diffs, min_val, max_val)
    processed_diffs = remove_high_intensity_pixels(normalized_diffs)
    min_val, max_val = find_min_max(processed_diffs)
    final_normalized_diffs = normalize_frames(processed_diffs, min_val, max_val)
    return final_normalized_diffs


if __name__ == "__main__":
    input_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/vid.avi'
    output_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/DIFFVideos_v4/v4_BN_output_video.avi'

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
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



