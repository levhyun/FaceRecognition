import os
import face_recognition
import cv2
import random

def exploreFacesTree():
    faces = {}
    faceDir = os.listdir('faces')
    for faceName in faceDir:
        images = os.listdir(f'faces/{faceName}')
        faces[faceName] = images
    return faces

def faceTreeShuffle(tree):
    for key in tree:
        random.shuffle(tree[key])
    return tree

def faceRecognition(image, tree):
    camImage = face_recognition.face_encodings(image)
    if len(camImage) == 0:
        return "Unknown"
    
    matched = {}
    for face in tree:
        cnt = 0
        for image in tree[face]:
            face_locations = face_recognition.load_image_file(f'faces/{face}/{image}')
            face_encodings = face_recognition.face_encodings(face_locations)
            if len(face_encodings) > 0:
                face_encodings = face_encodings[0]
                result = face_recognition.compare_faces(camImage, face_encodings)
                if result[0]:
                    print(f'Mached : {image}')
                    cnt += 1
        matched[face] = cnt
    maxKey = max(matched, key=matched.get)
    if matched[maxKey] > 2:
        return maxKey
    else:
        return 'Unknown'

cap = cv2.VideoCapture(0)
faceTree = exploreFacesTree()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text = faceRecognition(rgbFrame, faceTreeShuffle(faceTree))

    face_locations = face_recognition.face_locations(rgbFrame)

    for (top, right, bottom, left) in face_locations:
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
