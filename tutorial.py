#https://www.youtube.com/watch?v=mCcPmlr7y3U&t=236s -- Video do hashtag treinamentos

import cv2
import mediapipe as mp

# Iniciar a webcam
webcam = cv2.VideoCapture(0)

# Iniciar o reconhecimento de rostos
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho_rosto = mp.solutions.drawing_utils

# Iniciar o reconhecimento de mãos
solucao_reconhecimento_mao = mp.solutions.hands
reconhecer_mao = solucao_reconhecimento_mao.Hands()
desenho_mao = mp.solutions.drawing_utils

while True:
    # Lê a imagem da webcam
    verificador, frame = webcam.read()
    if not verificador:
        break

    # Reconhecimento de rostos
    listas_de_rostos = reconhecedor_rostos.process(frame)

    if listas_de_rostos.detections:
        for rosto in listas_de_rostos.detections:
            desenho_rosto.draw_detection(frame, rosto)

    # Reconhecimento de mãos
    listas_de_mao = reconhecer_mao.process(frame)

    if listas_de_mao.multi_hand_landmarks:
        for mao in listas_de_mao.multi_hand_landmarks:
            desenho_mao.draw_landmarks(frame, mao, solucao_reconhecimento_mao.HAND_CONNECTIONS)
            for landmark in mao.landmark:
                x, y, z = landmark.x, landmark.y, landmark.z

    # Mostra a imagem da webcam
    cv2.imshow("Rostos e Mãos", frame)

    # Fecha o aplicativo quando a tecla Esc é pressionada
    if cv2.waitKey(5) == 27:
        break

# Libera a webcam e fecha as janelas
webcam.release()
cv2.destroyAllWindows()
