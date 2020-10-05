import cv2, time
import numpy as np
import dlib
from scipy.spatial import distance as dist
import face_recognition

from math import hypot


def get_blinking_ratio(eye_points, landmarks):
    lp = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    rp = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)

    cp = (int((landmarks.part(eye_points[1]).x + landmarks.part(eye_points[2]).x) / 2),
          int((landmarks.part(eye_points[1]).y + landmarks.part(eye_points[2]).y) / 2))
    cd = (int((landmarks.part(eye_points[5]).x + landmarks.part(eye_points[4]).x) / 2),
          int((landmarks.part(eye_points[5]).y + landmarks.part(eye_points[4]).y) / 2))
    # notes eye region
    horizontal_line = cv2.line(frame, lp, rp, (0, 255, 0), 2)
    vertical_line = cv2.line(frame, cp, cd, (0, 255, 0), 2)

    hor_line_length = hypot((lp[0] - rp[0]), (lp[1] - rp[1]))
    ver_line_length = hypot((cp[0] - cd[0]), (cp[1] - cd[1]))
    # ratio = hor_line_length / ver_line_length
    ratio = ver_line_length / hor_line_length

    return ratio


def get_mouth_ratio(mouth_points, landmarks):
    rmp = (landmarks.part(mouth_points[0]).x, landmarks.part(mouth_points[0]).y)
    lmp = (landmarks.part(mouth_points[1]).x, landmarks.part(mouth_points[1]).y)
    umpc = (landmarks.part(mouth_points[2]).x, landmarks.part(mouth_points[2]).y)
    bmpc = (landmarks.part(mouth_points[3]).x, landmarks.part(mouth_points[3]).y)
    umpl = (landmarks.part(mouth_points[4]).x, landmarks.part(mouth_points[4]).y)
    bmpl = (landmarks.part(mouth_points[5]).x, landmarks.part(mouth_points[5]).y)
    umpr = (landmarks.part(mouth_points[6]).x, landmarks.part(mouth_points[6]).y)
    bmpr = (landmarks.part(mouth_points[7]).x, landmarks.part(mouth_points[7]).y)
    # notes mouth region
    horizontal_line = cv2.line(frame, rmp, lmp, (0, 255, 0), 2)

    vertical_line = cv2.line(frame, umpc, bmpc, (0, 255, 0), 2)
    vertical_line = cv2.line(frame, umpl, bmpl, (0, 255, 0), 2)
    vertical_line = cv2.line(frame, umpr, bmpr, (0, 255, 0), 2)

    hor_line_length = hypot((rmp[0] - lmp[0]), (rmp[1] - lmp[1]))

    ver_line_lengthc = hypot((umpc[0] - bmpc[0]), (umpc[1] - bmpc[1]))
    ver_line_lengthl = hypot((umpl[0] - bmpl[0]), (umpl[1] - bmpl[1]))
    ver_line_lengthr = hypot((umpr[0] - bmpr[0]), (umpr[1] - bmpr[1]))

    ratio = (ver_line_lengthc + ver_line_lengthl + ver_line_lengthr) / (3 * hor_line_length)

    return ratio

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

i = 0
flag = False
EYE_AR_CONSEC_FRAMES=50

predictorrepo = "shape_predictor_68_face_landmarks.dat"

imagerepo = "photo.jpg"
video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictorrepo)

font = cv2.FONT_HERSHEY_COMPLEX

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

counter_closed_eyes = 0
counter_open_mouth=0

image = face_recognition.load_image_file(imagerepo)
face_locations = face_recognition.face_locations(image)
face_landmarks_list = face_recognition.face_landmarks(image)

# known_image = face_recognition.load_image_file("biden.jpg")
# unknown_image = face_recognition.load_image_file("unknown.jpg")

try:
    iro_encoding = face_recognition.face_encodings(image)[0]
except:
    print("An error occured")

# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
i = 0
while True:
    i += 1
    check, frame = video.read()
    if i == 1:
        new_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        new_frame = new_frame[:, :, ::-1]

        frame_locations = face_recognition.face_locations(new_frame)
        try:

            frame_encodings = face_recognition.face_encodings(new_frame)
            results = face_recognition.face_distance(iro_encoding, frame_encodings)
            if (results < 0.6):
                print(results)
                time.sleep(5)
                cv2.putText(frame, "Hey", (50, 150), font, 7, (255, 0, 0))
                # print("Geia sou Iro")
        except:
            print("Error")

        # print("Spiros encodings ", spiros_encoding, " Frame encodings ", frame_encodings)


        # print(results)

    # break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        COUNTER += 1
        x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y) (x1, y1) (0,255,0), 2)
        landmarks = predictor(gray, face)
        # added for face detection representation
        landmarks2 = shape_to_np(landmarks)
        for (x, y) in landmarks2:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)

        mouth_ratio = get_mouth_ratio([48, 54, 51, 57, 50, 58, 52, 56], landmarks)
        # print((right_eye_ratio + left_eye_ratio) / 2.0)
        if ((right_eye_ratio + left_eye_ratio) / 2.0) < 0.2:
            cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

        # cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)
        #
        leregion = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                             (landmarks.part(37).x, landmarks.part(37).y),
                             (landmarks.part(38).x, landmarks.part(38).y),
                             (landmarks.part(39).x, landmarks.part(39).y),
                             (landmarks.part(40).x, landmarks.part(40).y),
                             (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # print((right_eye_ratio+left_eye_ratio)/2.0)
        if ((right_eye_ratio + left_eye_ratio) / 2.0) < 0.2:
            counter_closed_eyes += 1

            # print("Frames with closed eyes", counter_closed_eyes)
        if mouth_ratio > 0.63:
            cv2.putText(frame, "Don't yawn", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            counter_open_mouth += 1

        if COUNTER == 50:
            # if the alarm is not onq, turn it on

            perclos = counter_closed_eyes / COUNTER
            mof = counter_open_mouth / COUNTER
            print("PERCLOS ", perclos, "Frames closed eyes ", counter_closed_eyes, "Mouth aspect ratio", mouth_ratio)
            print("MOF ", mof, "Frames closed eyes ", counter_open_mouth, "Mouth aspect ratio", mouth_ratio)
            COUNTER = 0
            counter_closed_eyes = 0
            counter_open_mouth = 0

            if perclos > 0.4:
                ALARM_ON = True

        if ALARM_ON == True:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        min_x = np.min(leregion[:, 0])
        max_x = np.max(leregion[:, 0])
        min_y = np.min(leregion[:, 1])
        max_y = np.max(leregion[:, 1])

        eye = frame[min_y: max_y, min_x: max_x]
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        eye = cv2.resize(eye, None, fx=5, fy=5)

        cv2.imshow("Eye", eye)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('opencv.jpg', frame)
    elif key == ord('g'):
        if i % 2 == 0:
            flag = True
            i = 0
        else:
            flag = False
        i = i + 1

    if flag == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capture", frame)

video.release()

cv2.destroyAllWindows()


