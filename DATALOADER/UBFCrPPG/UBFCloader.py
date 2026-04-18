import os

def find_avi_files(main_directory):

    avi_files = []

    # 디렉토리를 순회하며 avi 파일을 검색
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path) and subdir.lower().startswith("subject"):
            for file in os.listdir(subdir_path):
                if file.endswith(".avi"):
                    file_path = os.path.join(subdir_path, file)
                    avi_files.append((file_path, subdir))  # 파일 경로와 디렉토리 이름을 튜플로 추가

    return avi_files

'''
# 메인 디렉토리 경로 설정
main_directory = '/media/neuroai/T7/rPPG/UBFC-rPPG/DATASET_2'

# avi 파일 찾기
avi_files = find_avi_files(main_directory)

# 찾은 avi 파일 출력
for avi_file in avi_files:
    print(avi_file)
'''