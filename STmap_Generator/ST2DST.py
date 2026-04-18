import os
from tqdm import tqdm
import cv2
import numpy as np


def find_image_files_by_dir(directory):
    image_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(root, file)
                subdir = os.path.relpath(root, directory)
                if subdir not in image_files:
                    image_files[subdir] = []
                image_files[subdir].append(file_path)
    return image_files


def process_image_for_column_difference(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    # 각 채널 분리
    b_channel, g_channel, r_channel = cv2.split(image)

    # 각 채널에 대해 열 차이 계산
    diff_b = np.zeros((b_channel.shape[0], b_channel.shape[1] - 1), dtype=np.int16)
    diff_g = np.zeros((g_channel.shape[0], g_channel.shape[1] - 1), dtype=np.int16)
    diff_r = np.zeros((r_channel.shape[0], r_channel.shape[1] - 1), dtype=np.int16)

    diff_b = b_channel[:, 1:] - b_channel[:, :-1]
    diff_g = g_channel[:, 1:] - g_channel[:, :-1]
    diff_r = r_channel[:, 1:] - r_channel[:, :-1]

    diff_image = cv2.merge((diff_b, diff_g, diff_r))
    cv2.imwrite(output_path, diff_image)


def process_images_in_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_path, file)

                try:
                    process_image_for_column_difference(input_file_path, output_file_path)
                    print(f"Processed and saved: {output_file_path}")
                except ValueError as e:
                    print(e)


if __name__ == "__main__":
    input_image_dir = '/home/neuroai/Projects/DSTMap/STMap/UBFC'  # 입력 이미지 디렉토리 경로 설정
    output_image_dir = '/home/neuroai/Projects/DSTMap/STMap/ST2DST'  # 출력 이미지 디렉토리 경로 설정

    if not os.path.exists(input_image_dir):
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_image_dir}")

    process_images_in_directory(input_image_dir, output_image_dir)
    print(f"모든 이미지를 처리하여 저장했습니다: {output_image_dir}")

