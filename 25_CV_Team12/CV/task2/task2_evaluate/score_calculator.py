import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

import onnxruntime

def save_scores_to_csv(results_list, final_avg, output_filename):
    """
    개별 파일의 점수와 최종 평균 점수를 CSV 파일로 저장합니다.
    """
    header = ['File_Name', 'Style_Similarity_Score']

    # 파일을 새로 생성합니다. (기존 데이터 덮어쓰기)
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for fname, score in results_list:
            writer.writerow([fname, f"{score:.4f}"])

        # 최종 평균 점수도 별도의 행으로 저장
        writer.writerow(['---', '---'])
        writer.writerow(['Average Score', f"{final_avg:.4f}"])

    print(f"\n Style 유사성 점수가 '{output_filename}'에 저장되었습니다.")


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, required=True)
    parser.add_argument('--style-folder', type=str, required=True)
    parser.add_argument('--output-csv', type=str, default='style_similarity_scores.csv', required=False)

    try:
        args = parser.parse_args()
    except SystemExit:
        print("Script definition successful. Please run the script in the next cell using the !python command.")
        exit()

    # check cmd arguments
    if not os.path.exists(args.result_folder):
        print('Cannot find the result folder: {0}'.format(args.result_folder))
        exit()
    if not os.path.exists(args.style_folder):
        print('Cannot find the style folder: {0}'.format(args.style_folder))
        exit()

    input_files = os.listdir(args.result_folder)
    style_files = os.listdir(args.style_folder)

    # 단일 스타일 파일 로드 및 고정 로직
    if len(style_files) != 1:
        print("Error: The style folder must contain exactly ONE target style image.")
        exit()

    single_style_fname = style_files[0]
    single_style_path = os.path.join(args.style_folder, single_style_fname)

    try:
        fixed_style_image = np.asarray(Image.open(single_style_path).convert('RGB').resize((512, 512))).astype(np.float32)
    except Exception as e:
        print(f"Error loading single style image at {single_style_path}: {e}")
        exit()

    # load the StyleSimiliaryDiscriminator model
    onnx_path = '/content/drive/MyDrive/task2_evaluate/StyleSimiliaryDiscriminator.onnx'
    try:
        onnx_sess = onnxruntime.InferenceSession(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model at {onnx_path}.")
        print(f"Detail: {e}")
        exit()


    print('--------------------------------------------------------------------------------')
    print('Result Folder: {0}'.format(args.result_folder))
    print('Target Style Image: {0}'.format(single_style_path))
    print('Total Result Images to process: {0}'.format(len(input_files)))
    print('--------------------------------------------------------------------------------')

    # 파일 이름과 점수를 함께 저장할 리스트
    results_list = []
    all_scores = [] # 평균 계산을 위한 리스트

    pbar = tqdm(input_files, total=len(input_files), unit='file')
    for idx, fname in enumerate(pbar):

        # load result image
        result_path = os.path.join(args.result_folder, fname)
        style_image = fixed_style_image # 고정된 스타일 이미지 사용

        try:
            result_image = np.asarray(Image.open(result_path).convert('RGB').resize((512, 512))).astype(np.float32)
        except Exception as e:
            pbar.write(f"\nSkipping file: {fname}. Error loading result image: {e}")
            continue

        # run onnx model
        score_output = onnx_sess.run(['score'], {'ref': style_image, 'img': result_image})
        score = score_output[0]

        # 0차원 배열에서 단일 스칼라 값 추출 (.item() 사용)
        try:
            single_score_value = score.item()
        except AttributeError:
            single_score_value = score[0] if isinstance(score, list) else score

        # 파일 이름과 점수를 리스트에 추가
        results_list.append((fname, single_score_value))
        all_scores.append(single_score_value) # 평균 계산 리스트에 추가

        # PBAR에 현재 처리 중인 파일의 점수를 표시
        pbar.set_description(f'Processing {fname} | Score: {single_score_value:.4f}')

    # 최종 평균 계산
    final_avg = np.mean(np.asarray(all_scores)) if all_scores else 0.0
    save_scores_to_csv(results_list, final_avg, args.output_csv)


    print('Individual Style Similiary Scores')


    for fname, score in results_list:
        print(f"  > File: {fname:<30} | Score: {score:.4f}")

    print('--------------------------------------------------------------------------------')
    print('Final Average Style Similiary Score: {0:.4f}'.format(final_avg))
    print('================================================================================')
