# import cv2 as cv
# import numpy as np
# from simple_facerec import SimpleFacerec

# sfr = SimpleFacerec()
# sfr.load_encoding_images("saveface/")

# cap = cv.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     # Detect Faces
#     face_locations, face_names = sfr.detect_known_faces(frame)
#     for face_loc, name in zip(face_locations, face_names):
#         y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

#         cv.putText(frame, name,(x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#         cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

#     cv.imshow("Frame", frame)

#     if cv.waitKey(1) == ord('q'):
#          break

# cap.release()
# cv.destroyAllWindows()


import cv2
import os

for filename in os.listdir("saveface"):
    foldername = "inputface"
    input_image = cv2.imread(foldername + "\\" + "OpecZ-2681.jpg")
    input_image = cv2.resize(input_image, (224, 224))
    
    image = cv2.imread(os.path.join("saveface", filename))
    image = cv2.resize(image, (224, 224))

    result = cv2.matchTemplate(image, input_image, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


    if min_val < 0.1:
        print(f'รูปภาพ {filename} ตรงกับ input (min_val = {min_val})')