import os
from tqdm import tqdm
import face_alignment
import torch
import cv2  # OpenCV를 임포트하여 BGR을 RGB로 변환
import STmap as su  # STmap 유틸리티 함수들을 임포트


def initialize_cuda():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA is available. Number of devices: {device_count}")
        else:
            print("CUDA is not available.")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")


def find_avi_files_by_dir(directory):
    avi_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".avi"):
                file_path = os.path.join(root, file)
                subdir = os.path.relpath(root, directory)
                if subdir not in avi_files:
                    avi_files[subdir] = []
                avi_files[subdir].append(file_path)
    return avi_files


def match_raw_and_diff_dirs(raw_files, diff_files):
    matched_files = []
    for subdir in raw_files:
        if subdir in diff_files:
            for raw_file, diff_file in zip(raw_files[subdir], diff_files[subdir]):
                matched_files.append((raw_file, diff_file, subdir))
    return matched_files


def is_image_rgb(image):
    """
    이미지가 RGB 형식인지 BGR 형식인지 확인
    """
    first_pixel = image[0, 0]
    if first_pixel[0] > first_pixel[2]:
        return False  # BGR 형식
    else:
        return True  # RGB 형식


def convert_bgr_to_rgb(image):
    """
    BGR 이미지를 RGB로 변환
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_video_for_stmap(raw_video_path, diff_video_path, subdir, output_stmap_root_path):
    try:
        input_filename = os.path.splitext(os.path.basename(raw_video_path))[0]
        output_dir = os.path.join(output_stmap_root_path, subdir)
        os.makedirs(output_dir, exist_ok=True)

        raw_frames = su.get_frames(raw_video_path)
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                          device='cuda' if torch.cuda.is_available() else 'cpu')
        lmks = su.get_landmarks(fa, raw_frames)
        diff_frames = su.get_frames(diff_video_path)
        aligned_faces = su.align_face(diff_frames, lmks)
        stmap_yuv = su.STmap(aligned_faces)

        # stmap_yuv에서 RGB 변환 시 RGB인지 확인
        stmap_rgb = su.YUV2RGB(stmap_yuv)
        if not is_image_rgb(stmap_rgb):
            stmap_rgb = convert_bgr_to_rgb(stmap_rgb)

        su.save_STmap(stmap_rgb, output_dir, f'{input_filename}_stmap_rgb.png', convert_to_bgr=False)
        print(f"Processed and saved STmaps (RGB) for {input_filename} in {subdir}")

        su.save_STmap(stmap_yuv, output_dir, f'{input_filename}_stmap_yuv.png')

    except Exception as e:
        print(f"Failed to process video {raw_video_path} and {diff_video_path} in {subdir}: {e}")


def create_stmap_for_videos(raw_vid_dir_path, diff_vid_dir_path, output_stmap_root_path):
    raw_files = find_avi_files_by_dir(raw_vid_dir_path)
    diff_files = find_avi_files_by_dir(diff_vid_dir_path)
    matched_files = match_raw_and_diff_dirs(raw_files, diff_files)

    if not matched_files:
        print("No matched files found. Please check the directory structure and file naming conventions.")
        return

    print(f"Found {len(matched_files)} matched files.")
    for raw_path, diff_path, subdir in tqdm(matched_files, desc="Creating STmaps"):
        tqdm.write(f"Processing Raw: {raw_path}, Diff: {diff_path}, Subdir: {subdir}")
        process_video_for_stmap(raw_path, diff_path, subdir, output_stmap_root_path)


if __name__ == "__main__":
    initialize_cuda()

    raw_vid_path = '/media/neuroai/T7/rPPG/UBFC-rPPG/DATASET_2'  # Path to raw video directory
    diff_vid_path = '/DATALOADER/UBFCrPPG/DIFFvideos_DVM7'  # Path to diff video directory
    output_stmap_path = '/home/neuroai/Projects/DSTMap/DSTMap/UBFC_DVM7'  # Path to output STmap directory

    create_stmap_for_videos(raw_vid_path, diff_vid_path, output_stmap_path)
