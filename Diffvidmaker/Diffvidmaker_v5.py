import cv2
import os
import numpy as np

def detect_and_crop_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = frame[y:y + h, x:x + w]
        return cropped_face
    else:
        return None

def compute_frame_difference(input_video_path, face_cascade_path):
    if not os.path.exists(input_video_path):
        print(f"Error: File {input_video_path} does not exist.")
        return []

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade.")
        return []

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frame_diffs = []
    prev_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_face = detect_and_crop_faces(frame, face_cascade)
        if curr_face is None:
            continue

        if prev_face is None:
            prev_face = curr_face
            continue

        if prev_face.shape == curr_face.shape:
            frame_diff = cv2.absdiff(prev_face, curr_face)
            frame_diffs.append(frame_diff)
            prev_face = curr_face

    cap.release()

    return frame_diffs

def find_min_max(frame_diffs):
    min_val = np.array([255, 255, 255])
    max_val = np.array([0, 0, 0])

    for diff in frame_diffs:
        b, g, r = cv2.split(diff)
        min_val = np.minimum(min_val, [np.min(b), np.min(g), np.min(r)])
        max_val = np.maximum(max_val, [np.max(b), np.max(g), np.max(r)])

    return min_val, max_val

def normalize_frames(frame_diffs, min_val, max_val):
    normalized_diffs = []
    for diff in frame_diffs:
        b, g, r = cv2.split(diff)

        b = ((b - min_val[0]) * 255.0 / (max_val[0] - min_val[0])).astype(np.uint8)
        g = ((g - min_val[1]) * 255.0 / (max_val[1] - min_val[1])).astype(np.uint8)
        r = ((r - min_val[2]) * 255.0 / (max_val[2] - min_val[2])).astype(np.uint8)

        normalized_diff = cv2.merge((b, g, r))
        normalized_diffs.append(normalized_diff)

    return normalized_diffs

def remove_high_intensity_pixels(frame_diffs, threshold=20):
    processed_diffs = []
    for diff in frame_diffs:
        mask = np.any(diff >= threshold, axis=-1)
        processed_diff = diff.copy()
        processed_diff[mask] = 0
        processed_diffs.append(processed_diff)
    return processed_diffs

def save_frame_differences(frame_diffs, output_video_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size, isColor=True)

    for diff in frame_diffs:
        out.write(diff)

    out.release()

input_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/vid.avi'
output_video_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Videos/DIFFVideos_v5/v5_BN_output_video.avi'
face_cascade_path = '/home/neuroai/Projects/DSTMap/Diffvidmaker/Haar/haarcascade_frontalface_default.xml'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    frame_diffs = compute_frame_difference(input_video_path, face_cascade_path)

    if frame_diffs:
        min_val, max_val = find_min_max(frame_diffs)
        normalized_diffs = normalize_frames(frame_diffs, min_val, max_val)

        processed_diffs = remove_high_intensity_pixels(normalized_diffs)

        min_val, max_val = find_min_max(processed_diffs)
        final_normalized_diffs = normalize_frames(processed_diffs, min_val, max_val)

        frame_size = (final_normalized_diffs[0].shape[1], final_normalized_diffs[0].shape[0])
        save_frame_differences(final_normalized_diffs, output_video_path, fps, frame_size)

