import os
import shutil

TEXT_EXTENSION = '.json'

def copy_txt_from_dst_foldernames(src_main_dir, dst_main_dir):
    # 각 대상 폴더명을 기준으로 .json 파일 매핑
    for dst_folder in os.listdir(dst_main_dir):
        dst_folder_path = os.path.join(dst_main_dir, dst_folder)

        # 폴더일 경우에만 처리
        if os.path.isdir(dst_folder_path):
            base_name = dst_folder.rsplit("_", 1)[0]  # 예: "Video1-1" → "Video1"
            txt_filename = f"{base_name}{TEXT_EXTENSION}"
            src_txt_path = os.path.join(src_main_dir, txt_filename)
            dst_txt_path = os.path.join(dst_folder_path, txt_filename)

            if os.path.exists(src_txt_path):
                shutil.copy2(src_txt_path, dst_txt_path)
                print(f'Copied {src_txt_path} to {dst_txt_path}')
            else:
                print(f'Warning: {src_txt_path} not found!')

# 디렉토리 설정
src_main_dir = '/media/neuroai/T7 Shield/Dataset/VitalVideos/vv250'
dst_main_dir = '/media/neuroai/T7/rPPG/STMap_raw/vv250'

# 복사 실행
copy_txt_from_dst_foldernames(src_main_dir, dst_main_dir)
