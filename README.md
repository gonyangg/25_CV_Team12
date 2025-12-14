# 25_CV_Team12
2025-가을 학기 CV 프로젝트

## demo.ipynb
포즈 & 색상에 대한 가이드를 주는 사진 가이드 시스템에 대한 데모 코드입니다. 
T4 gpu를 필요로 하거나 colab 파일에서 설정할 수 없는 코드는 아래의 3d_mesh.py, webcam.py로 실행할 수 있도록 하였습니다. 

## 3d_mesh.py
SAM3D_Body 모델을 이용해 3d mesh 이미지를 저장하는 코드입니다.
1. 환경을 설정하고 활성화하세요
   ```
   conda create -n 3d_mesh python=3.11 -y
   conda activate 3d_mesh
   ```
2. requirement.txt를 이용해 환경 설정을 해주세요.
   ```
   pip install -r requirements.txt --no-build-isolation
   ```
3. detectron 2를 설치하세요
   ```
   pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
   ```
4. MoGe를 설치하세요
   ```
   pip install git+https://github.com/microsoft/MoGe.git
   ```

## webcam.py
3d mesh 가이드를 웹캠에 띄워 포즈 가이드를 주고, 찍은 사진을 저장하는 코드입니다. 

[코드 실행 key]

s- 사진 촬영

ESC- 프로그램 중단
