import Diffvidmaker_v7 as DVM7
'''
add here new version of Diffvidmaker.py
'''

import sys
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

main_dir = os.path.join(os.path.dirname(__file__), '..', 'UBFCrPPG')
sys.path.append(main_dir)

import DATALOADER.UBFCrPPG.UBFCloader as UL
'''
add here new datasetloader.py
'''

def DVMsample(vid_dir_path, output_vid_root_path):
    files = UL.find_avi_files(vid_dir_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(lambda f: process_video(f[0], f[1], output_vid_root_path), files), total=len(files), desc="Making diff videos"))

def process_video(input_video_path, subdir, output_vid_root_path):
    try:
        input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
        output_dir = os.path.join(output_vid_root_path, subdir)
        os.makedirs(output_dir, exist_ok=True)
        output_video_file = os.path.join(output_dir, f'{input_filename}_diff.avi')

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        cap.release()

        frame_diffs = DVM7.compute_frame_difference(input_video_path)
        if frame_diffs:
            processed_diffs = DVM7.normalize_and_process_frames(frame_diffs)
            DVM7.save_frame_differences(processed_diffs, output_video_file, fps, frame_size)
        else:
            print(f"No frame differences computed for video file {input_video_path}")

    except Exception as e:
        print(f"Error processing video file {input_video_path}: {e}")

if __name__ == '__main__':
    input_vid = '/media/neuroai/T7/rPPG/UBFC-rPPG/DATASET_2'
    output_vid = '/home/neuroai/Projects/DSTMap/UBFCrPPG/DIFFvideos_DVM7'

    DVMsample(input_vid, output_vid)
