import os
import shutil

def copy_txt_files(src_main_dir, dst_main_dir):
    for src_dirpath, _, filenames in os.walk(src_main_dir):
        # 대상 디렉토리 경로 계산
        relative_path = os.path.relpath(src_dirpath, src_main_dir)
        dst_dirpath = os.path.join(dst_main_dir, relative_path)

        # 대상 디렉토리 생성
        os.makedirs(dst_dirpath, exist_ok=True)

        # .txt 파일만 복사
        for filename in filenames:
            if filename.endswith('.txt'):  # TXT 파일만 필터링
                src_file = os.path.join(src_dirpath, filename)
                dst_file = os.path.join(dst_dirpath, filename)
                shutil.copy2(src_file, dst_file)
                print(f'Copied {src_file} to {dst_file}')

# 소스 및 대상 디렉토리 설정
src_main_dir = '/media/neuroai/T7/rPPG/UBFC-rPPG/DATASET_2'
dst_main_dir = '/media/neuroai/T7/rPPG/STMap_raw/UBFC'

# 복사 실행
copy_txt_files(src_main_dir, dst_main_dir)
