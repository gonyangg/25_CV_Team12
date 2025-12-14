import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity
import csv # <--- CSV 라이브러리 추가

import torch
from torchvision import transforms

import ldc 

# ----------------------------------------------------
# LDC_PTH_PATH: 복사된 파일 경로
# ----------------------------------------------------
LDC_PTH_PATH = './ldc.pth'

# [핵심 수정]: 모든 이미지의 크기를 통일하기 위한 전역 변수
FIXED_H = 0
FIXED_W = 0
device = torch.device("cpu") 

# LDC 모델 로드 및 평가 모드 설정
try:
    ldc_model = ldc.LDC()
    ldc_model.load_state_dict(torch.load(LDC_PTH_PATH, map_location=device))
    ldc_model.to(device).eval()
except Exception as e:
    print("\n[FATAL ERROR] LDC 모델 로드 실패. 경로와 파일 존재 여부를 확인하세요.")
    print(f"경로: {LDC_PTH_PATH}")
    print(f"세부 오류: {e}")
    exit()

# ----------------------------------------------------
# [새로 추가된 함수] CSV 파일 저장 함수
# ----------------------------------------------------
def save_scores_to_csv(results_list, final_avg, output_filename):
    """
    개별 파일의 점수와 최종 평균 점수를 CSV 파일로 저장합니다.
    """
    header = ['File_Name', 'Content_Similarity_Score']
    
    # 파일을 새로 생성합니다. (기존 데이터 덮어쓰기)
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for fname, score in results_list:
            writer.writerow([fname, f"{score:.4f}"])
            
        # 최종 평균 점수도 별도의 행으로 저장
        writer.writerow(['---', '---'])
        writer.writerow(['Average Score', f"{final_avg:.4f}"])
            
    print(f"\n✅ Content 유사성 점수가 '{output_filename}'에 저장되었습니다.")


def calculate_ldc_edge(image_path, is_fixed_content=False):
    global FIXED_H, FIXED_W
    
    image = Image.open(image_path).convert('RGB')

    h_orig, w_orig = image.size
    
    # LDC 모델 입력 요구 사항: 32의 배수로 크기 조정
    h_mod32 = int(h_orig - h_orig % 32)
    w_mod32 = int(w_orig - w_orig % 32)
    
    if h_mod32 == 0 or w_mod32 == 0:
        raise ValueError(f"Image size is too small after mod-32 adjustment: {h_mod32}x{w_mod32}")

    
    # [크기 고정 로직]
    if is_fixed_content:
        FIXED_H = h_mod32
        FIXED_W = w_mod32
        h, w = h_mod32, w_mod32
    elif FIXED_H != 0 and FIXED_W != 0:
        h, w = FIXED_H, FIXED_W
    else:
        h, w = h_mod32, w_mod32 
        
    
    # 이미지 전처리 및 텐서 변환
    mean = torch.tensor([103.939, 116.779, 123.68]).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    # 모든 이미지를 FIXED_H x FIXED_W 크기로 조정하여 전달합니다.
    image = transforms.functional.resize(image, (h, w))
    image = transforms.functional.to_tensor(image)[None, ...].to(device) * 255

    with torch.no_grad():
        edges = ldc_model(image - mean)
    
    avg_edge = ldc.postprocess_edges(edges)
    avg_edge = torch.from_numpy(avg_edge).unsqueeze(0).unsqueeze(0) / 255

    return avg_edge


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, required=True)
    parser.add_argument('--content-folder', type=str, required=True)
    parser.add_argument('--output-csv', type=str, default='content_similarity_scores.csv', required=False) # <--- CSV 출력 파일명 인자 추가
    
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Parsing arguments failed. Please run the script in the next cell using the !python command.")
        exit()

    # check cmd arguments
    if not os.path.exists(args.result_folder):
        print('Cannot find the result folder: {0}'.format(args.result_folder))
        exit()
    if not os.path.exists(args.content_folder):
        print('Cannot find the content folder: {0}'.format(args.content_folder))
        exit()

    result_files = os.listdir(args.result_folder)
    content_files = os.listdir(args.content_folder)
    
    if not result_files or not content_files:
        print("\n[ERROR] One or both folders are EMPTY. Please check your image files.")
        exit()
        
    # [단일 원본 파일 로드 및 고정 로직]
    if len(content_files) != 1:
        print(f"\n[ERROR] Content folder must contain exactly ONE original image, found {len(content_files)}.")
        exit()
        
    single_content_fname = content_files[0]
    single_content_path = os.path.join(args.content_folder, single_content_fname)
    
    try:
        # ❗ 첫 번째 호출: 고정된 원본 이미지의 에지 추출 및 크기 고정 (is_fixed_content=True)
        fixed_content_edge = calculate_ldc_edge(single_content_path, is_fixed_content=True) 
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to process the single content image: {e}")
        exit()
    # ----------------------------------------------------


    print('--------------------------------------------------------------------------------')
    print('Result Folder: {0}'.format(args.result_folder))
    print('Fixed Original Image: {0}'.format(single_content_path))
    print('Fixed Image Size (HxW): {0}x{1}'.format(FIXED_H, FIXED_W))
    print('Total Result Images to process: {0}'.format(len(result_files)))
    print('Processing Device: {0}'.format(device.type))
    print('--------------------------------------------------------------------------------')

    # calculate content ssim score
    results_list = []
    all_scores = []
    
    pbar = tqdm(result_files, total=len(result_files), unit='file')
    for idx, fname in enumerate(pbar):

        result_path = os.path.join(args.result_folder, fname)
        content_edge = fixed_content_edge # 고정된 원본 에지 사용
        
        try:
            # ❗ 두 번째 호출: 결과 이미지의 에지 추출 (고정된 크기 사용)
            result_edge = calculate_ldc_edge(result_path) 
        except Exception as e:
            pbar.write(f"\n[SKIP] File {fname} failed LDC processing: {e}")
            continue

        # 2. NumPy 배열로 변환
        result_edge = result_edge[0].permute(1, 2, 0).cpu().numpy()
        content_edge = content_edge[0].permute(1, 2, 0).cpu().numpy()
        
        # 3. SSIM 점수 계산
        score = structural_similarity(result_edge, content_edge, channel_axis=-1, data_range=1.0)
        
        results_list.append((fname, score)) 
        all_scores.append(score)
        pbar.set_description(f'Processing {fname} | Score: {score:.4f}') 


    # 최종 결과 출력 및 CSV 저장
    final_avg = np.mean(np.asarray(all_scores)) if all_scores else 0.0
    
    # ----------------------------------------------------
    # [새로 추가된 코드] CSV 파일 저장 호출
    # ----------------------------------------------------
    save_scores_to_csv(results_list, final_avg, args.output_csv)
    
    
    print('\n================================================================================')
    print('                🌟 Individual Content Similiary Scores 🌟')
    print('================================================================================')
    
    for fname, score in results_list:
        print(f"  > File: {fname:<30} | Score: {score:.4f}")

    print('--------------------------------------------------------------------------------')
    print('Final Average Content Similiary Score: {0:.4f}'.format(final_avg))
    print('================================================================================')
