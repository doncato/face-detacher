# By doncato <https://github.com/doncato>
# Licensed under the GNU General Public License version 3.0

## Description: The main goal of this Programm is to automatically detect and
## annonymize faces in pictures. This may or may not work and may or may not be
## useful. This code is not known to cause cancer. You are free to take selfies
## with this programm and share these with your friends, otherwise you are free
## to do whatever you want.

## Dependencies:
## You need python3 (duh)
## You need to install the following libraries:
## + OpenCV

# Default libraries
import sys
# Additional libraries
import cv2
import numpy as np

cascades = (
    'data/haarcascade_frontalface_default.xml',
    'data/haarcascade_eye.xml',
    'data/',
)


def get_envargs():
    image_path = sys.argv[1]
    return image_path


class censor_handling():
    def __init__(self):
        self.scales = [1.1, 1.25, 1.5, 2]

    def get_objs(self, gray_img, cascade):
        res = np.array(np.empty(shape=(0, 4), dtype='i'))
        for scale in self.scales:
            objs = cascade.detectMultiScale(
                gray_img,
                scaleFactor=scale,
                minNeighbors=5,
                minSize=(64, 64),
            )
            if type(objs) == np.ndarray and objs.ndim == res.ndim:
                res = np.append(res, objs, axis=0)
        return res

def rect_contains(rect, q):
    (x, y, h, w) = rect
    (x_q, y_q, h_q, w_q) = q
    q_center = (x + w/2, y+ h/2)
    if (
        q_center[0] > x
        and q_center[0] < (x+w)
        and q_center[1] > y
        and q_center[1] < (y+h)
    ):
        return true
    return false


def remove_contained_rects(rectangles):
    important_rects = rectangles.tolist()
    for rect in rectangles:
        for other in rectangles:
            if rect_contains(rect, other):
                important_rects.remove(other)

    return important_rects


def censor_eyes(eyes, faces):
        eye_pairs = []
        for face in faces:
            (x_f, y_f, w_f, h_f) = face
            eye_pair = []
            for eye in eyes:
                if rect_contains(face, eye):
                    eye_pair.append(eye)
                    continue

            if len(eye_pair) > 0:
                eye_pairs.append(((x_f, y_f, w_f, h_f), eye_pair))

        rectangles = []
        for eye_pair in eye_pairs:
            x = eye_pair[0][0]
            w = eye_pair[0][0] + eye_pair[0][2]
            rect = None
            for (x_e, y_e, w_e, h_e) in eye_pair[1]:
                if rect is None:
                    rect = [y_e, y_e+h_e]
                    continue

                if y_e < rect[0]:
                    rect[0] = y_e
                if (y_e+h_e) > rect[1]:
                    rect[1] = y_e+h_e

            rectangles.append((x, rect[0], w, rect[1]))

        return rectangles


def main():
    args = get_envargs()
    faceCascade = cv2.CascadeClassifier(cascades[0])
    eyeCascade = cv2.CascadeClassifier(cascades[1])
    #print(faceCascade.empty())

    image = cv2.imread(args)
    og_image = cv2.imread(args)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    handler = censor_handling()

    faces = handler.get_objs(gray, faceCascade)
    eyes = handler.get_objs(gray, eyeCascade)

    print(f"{len(faces)}/{len(eyes)} Faces/Eyes detected")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)

    for (x, y, a, b) in handler.censor_eyes(eyes, faces):
        cv2.rectangle(og_image, (x,y), (a,b), (0,0,0,), -1)


    cv2.imwrite("output/debug_out.png", image)
    cv2.imwrite("output/out.png", og_image)

if __name__ == "__main__":
    main()
