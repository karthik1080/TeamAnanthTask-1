import numpy as np
import cv2


def crop(fac, img):
    y = len(img)
    x = len(img[0])
    while y % fac != 0:
        y -= 1
    while x % fac != 0:
        x -= 1
    return img[:y, :x]


def binn(fac, img):
    ch = cv2.split(crop(fac, img))
    ch = list(ch)
    for c in range(0, len(ch)):
        temp = np.zeros((int(len(ch[c])/fac), int(len(ch[c][0])/fac)))
        for i in range(0, fac):
            for j in range(0, fac):
                temp += ch[c][i::fac, j::fac]
        temp = np.divide(temp, (fac**2)*255)
        ch[c] = temp
    return cv2.merge((ch[0], ch[1], ch[2]))


img = cv2.imread("clock_tower.jpg")
cv2.imshow("clock_tower.jpg",img)
img = binn(4, img)
cv2.imshow("pog", img)
cv2.waitKey(0)