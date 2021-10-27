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

def get_envargs():
    image_path = sys.argv[1]
    casc_path = sys.argv[2]
    return (image_path, casc_path)


def main():
    args = get_envargs()
    faceCascade = cv2.CascadeClassifier(args[1])
    print(faceCascade.empty())

    image = cv2.imread(args[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    print(f"{len(faces)} Faces detected")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()
