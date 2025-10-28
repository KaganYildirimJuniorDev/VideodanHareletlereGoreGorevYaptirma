import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands

hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

engine = pyttsx3.init()

checkThumbsUp = False
checkHandsUp = False

while True:
    ret, frame = cam.read()
    if not ret:
        print("Kamera okunamadı.")
        break

    if checkThumbsUp:
        engine.say("Thumbs Up detected")
        engine.runAndWait()
        checkThumbsUp = False

    if checkHandsUp:
        engine.say("Eller Yukari")
        engine.runAndWait()
        checkHandsUp = False
    
    camRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hlms = hands.process(camRGB)

    height, width, _ = frame.shape
    
    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:
            
            for fingernum, landmark in enumerate(handlandmarks.landmark):
                positionX = int(landmark.x * width)
                positionY = int(landmark.y * height)
                cv2.putText(frame, str(fingernum), (positionX, positionY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            try:
                lm = handlandmarks.landmark
                
                thumb_tip_y = lm[4].y
                thumb_ip_y = lm[3].y
                
                index_tip_y = lm[8].y
                index_pip_y = lm[6].y
                
                middle_tip_y = lm[12].y
                middle_pip_y = lm[10].y
                
                ring_tip_y = lm[16].y
                ring_pip_y = lm[14].y
                
                pinky_tip_y = lm[20].y
                pinky_pip_y = lm[18].y


                if (thumb_tip_y < thumb_ip_y and
                    index_tip_y < index_pip_y and
                    middle_tip_y < middle_pip_y and
                    ring_tip_y < ring_pip_y and
                    pinky_tip_y < pinky_pip_y):
                    
                    print("ELLER YUKARI!")
                    cv2.putText(frame, "ELLER YUKARI", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    checkHandsUp = True

       
       
                elif (thumb_tip_y < thumb_ip_y and      
                      index_tip_y > index_pip_y and     
                      middle_tip_y > middle_pip_y and   
                      ring_tip_y > ring_pip_y and       
                      pinky_tip_y > pinky_pip_y):       
                    
                    print("THUMBS UP!")
                    cv2.putText(frame, "THUMBS UP", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    checkThumbsUp = True

            except Exception as e:
                print(f"Jest algılama hatası: {e}")

            mpDraw.draw_landmarks(frame, handlandmarks, mphands.HAND_CONNECTIONS)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()