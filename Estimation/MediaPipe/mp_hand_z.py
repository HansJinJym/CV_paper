import cv2
import mediapipe as mp
from tqdm import tqdm
import time

# 导入solution
mp_hands = mp.solutions.hands
# 导入模型
hands = mp_hands.Hands(
    static_image_mode=False,            # 静态图or连续帧
    max_num_hands=1,                    # 最多检测手数量
    min_detection_confidence=0.7,       # 置信度阈值
    min_tracking_confidence=0.5)        # 追踪阈值
# 绘图函数
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_frame(img):
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    img = cv2.flip(img, 1)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)

    if results.multi_hand_landmarks:

        handness_str = ' '
        index_finger_tip_str = ' '
        hand_area = ' '

        for hand_idx in range(len(results.multi_hand_landmarks)):
            # 手21关键点
            hand_21 = results.multi_hand_landmarks[hand_idx]
            # 关键点连线
            mpDraw.draw_landmarks(
                img, hand_21, mp_hands.HAND_CONNECTIONS,
                #mp_drawing_styles.get_default_hand_landmarks_style(),
                #mp_drawing_styles.get_default_hand_connections_style()
                )
            # 左右手信息
            temp_handness = results.multi_handedness[hand_idx].classification[0].label
            handness_str += '{}:{} '.format(hand_idx, temp_handness)
            
            cz0 = hand_21.landmark[0].z
            for i in range(21):
                cx = int(hand_21.landmark[i].x * w)
                cy = int(hand_21.landmark[i].y * h)
                cz = hand_21.landmark[i].z
                depth_z = cz0 - cz

                radius = max(int(6 * (1 + depth_z*5)), 0)

                img = cv2.circle(img, (cx, cy), radius, (0,255,255), -1)

            ########### kernel has been changed ###########
            if results.hand_rects:
                rect = results.hand_rects[hand_idx]
                rx, ry, rh, rw = rect.x_center*img.shape[1], rect.y_center*img.shape[0], rect.height*img.shape[0], rect.width*img.shape[1]
                cv2.rectangle(img, (int(rx-rw/3),int(ry-rh/3)), (int(rx+rw/3),int(ry+rh/3)), (255,0,0), 2)
                area = str(int(rh*rw/100))
                hand_area += '{}:{} '.format(hand_idx, area)

            # if results.palm_detections:
            #     print(results.palm_detections)
            #     #palm = results.palm_detections[hand_idx]
            #     # print(palm.location_data.relative_bounding_box)
            #     #mpDraw.draw_detection(img, palm)
            ###############################################
        if results.palm_detections:
            print(results.palm_detections)
            for palm in results.palm_detections:
                # print(palm.location_data.relative_bounding_box)
                mpDraw.draw_detection(img, palm)

        scalar = 1
        img = cv2.putText(img, handness_str, (25*scalar, 100*scalar), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scalar, (255,0,255), 2*scalar)
        img = cv2.putText(img, index_finger_tip_str, (25*scalar, 150*scalar), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scalar,(255,0,255), 2*scalar)

        fps = 1/(time.time()-start_time)
        img = cv2.putText(img, 'fps: '+str(int(fps)), (25*scalar, 150*scalar), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scalar,(255,0,255), 2*scalar)

        # img = cv2.putText(img, 'z0: '+str(cz0), (25*scalar, 200*scalar), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scalar,(255,0,255), 2*scalar)

        img = cv2.putText(img, 'S_hand: '+hand_area, (25*scalar, 200*scalar), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scalar,(255,0,255), 2*scalar)
        
    
    return img

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.open(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        cv2.imshow('mp_hand_z', frame)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()