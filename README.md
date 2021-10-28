# Face-Detacher
Quickly 'anonymize' all people in an image. This script will put a black bar over all eye-pairs in an image

This is a small python script to detect and censor all eyepairs in an image.

Feel free to use this script if you think it's helpful (and feel free to attribute me).

You can also contribute to this by forking this repository and making a pull request if you want to help.

Depending on the Image this programm may work differently. Especially partially cut of faces, faces at an angle and people looking
sideways are tough to detect.

## How2use:
1. You need python3 installed
2. You need to have the default python libraries as well as pip installed
3. You need the libraries numpy and cv2 to install those use `pip3 install opencv-python numpy`
4. open your terminal/cmd and change your working directory to the directory of the script
5. type `python3 main.py <path/to/image>` and hit enter
