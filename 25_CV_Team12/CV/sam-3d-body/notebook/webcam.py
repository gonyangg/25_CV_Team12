import cv2
import numpy as np

#==========================
#경로 확인 부탁드립니다
#
PROJECT_ROOT="C:/Users/Downloads/CV/sam-3d-body/notebook"
#==========================


ORIGINAL_PATH = f"{PROJECT_ROOT}/images/original2.jpg"

orig_img = cv2.imread(ORIGINAL_PATH)
assert orig_img is not None, "Failed to load original image"

MESH_PATH = f"{PROJECT_ROOT}/output/original2/original2_overlay_000.png"   
W_REF = orig_img.shape[1]            # 원본 이미지 width
H_REF = orig_img.shape[0]            # 원본 이미지 height
CAMERA_ID = 0

# =========================
# RGBA overlay 함수
# =========================
def overlay_rgba(bg, fg, x, y, alpha_scale):
    h, w = fg.shape[:2]

    # 클리핑 영역 계산
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, bg.shape[1])
    y2 = min(y + h, bg.shape[0])

    if x1 >= x2 or y1 >= y2:
        return bg

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    alpha = fg[fg_y1:fg_y2, fg_x1:fg_x2, 3] / 255.0 * alpha_scale

    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            (1 - alpha) * bg[y1:y2, x1:x2, c]
            + alpha * fg[fg_y1:fg_y2, fg_x1:fg_x2, c]
        )

    return bg


# =========================
# Load mesh PNG
# =========================
mesh_png = cv2.imread(MESH_PATH, cv2.IMREAD_UNCHANGED)
assert mesh_png is not None, "Failed to load mesh.png"

# RGB → RGBA 변환
if mesh_png.shape[2] == 3:
    gray = cv2.cvtColor(mesh_png, cv2.COLOR_BGR2GRAY)

    # 흰 배경 제거 
    alpha = (gray < 250).astype(np.uint8) * 255

    mesh_png = np.dstack([mesh_png, alpha])


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(CAMERA_ID)
assert cap.isOpened(), "Cannot open webcam"

while True:
    ret, frame_raw = cap.read()
    if not ret:
        break
    frame_vis = frame_raw.copy()
    # ---- bbox from detector (원본 기준) ----
    x1, y1, x2, y2 =314.85995, 186.61899, 691.1285,  750.71356
    #208.77287, 239.15314, 659.6683,  852.7657#192.53389, 120.43255, 621.80853, 714.8016
    # #314.85995, 186.61899, 691.1285,  750.71356#207.93924, 164.47025, 830.5962,  998.79944
    bw = x2 - x1
    bh = y2 - y1

    # ---- webcam frame size ----
    H_cam, W_cam = frame_raw.shape[:2]

    # ---- 원하는 bbox 크기 (웹캠 기준) ----
    target_box_h = int(H_cam * 0.7)
    target_box_w = int(target_box_h * (bw / bh))

    # ---- scale ----
    scale = target_box_h / bh

    # ---- resize mesh ----
    mesh_scaled = cv2.resize(
        mesh_png,
        (int(W_REF * scale), int(H_REF * scale)),
        interpolation=cv2.INTER_LINEAR
    )


    bbox_cx_ref = (x1 + x2) / 2
    bbox_cy_ref = (y1 + y2) / 2

    rel_cx = bbox_cx_ref / W_REF
    rel_cy = bbox_cy_ref / H_REF

    cx_cam = rel_cx * W_cam
    cy_cam = rel_cy * H_cam


    offset_x = int(cx_cam - scale * bbox_cx_ref)
    offset_y = int(cy_cam - scale * bbox_cy_ref)


    # ---- overlay ----
    frame_vis = overlay_rgba(
        frame_vis,
        mesh_scaled,
        offset_x,
        offset_y,
        alpha_scale=0.7
    )

    h_cam = frame_vis.shape[0]
    h_ori, w_ori = orig_img.shape[:2]

    scale_ori = h_cam / h_ori
    orig_resized = cv2.resize(
        orig_img,
        (int(w_ori * scale_ori), h_cam),
        interpolation=cv2.INTER_LINEAR
    )

    # 2) 좌우로 붙이기
    combined = np.hstack([frame_vis, orig_resized])

    # 3) 표시
    cv2.imshow("Pose Guide + Original", combined)

    key = cv2.waitKey(1) & 0xFF #ESC로 off
    if key == 27:
        break
    elif key == ord("s"):
        cv2.imwrite(f"{PROJECT_ROOT}/webcam_results/capture.jpg", frame_raw)
        print("Saved capture.jpg (guide not included)")

cap.release()
cv2.destroyAllWindows()