import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import moviepy.editor as mpe
import time
import pandas as pd
import string

gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~i!lI;:,\"^`'. "
gscale2 = "@%#*+=-:. "
chars = ""

charDimension = 40
grayChars = pd.read_csv("grayChars.csv", index_col=0)

def getCost(char, valueList):
    charList = np.array(grayChars.loc[char])
    listDiff = np.subtract(valueList, charList)
    cost = np.sum(np.square(listDiff))
    return cost
def getMinCost(valueList):
    chars = string.printable
    chars = chars.strip() + " "
    currentCost = getCost("0", valueList)
    currentChar = ""
    for char in chars:
        evalCost = getCost(char, valueList)
        if evalCost < currentCost:
            currentCost = evalCost
            currentChar = char
    return currentChar

def cvt_ascii(img):
    H, W = img.shape[0], img.shape[1]
    w = charDimension
    h = charDimension
    rows = int(H/h)
    cols = int(W/w)
    ascii_img = []
    for j in range(rows):
        y1 = int(j*h)
        y2 = int((j+1)*h)
        if j == rows - 1:
            y2 = H
        ascii_img.append("")
        for i in range(cols):
            x1 = int(i * w)
            x2 = int((i + 1) * w)
            if i == cols - 1:
                x2 = W
            cropped = img[y1:y2, x1:x2]
            grayScaleChunk = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            valueList = np.array(np.array(grayScaleChunk).flat)
            gsval = getMinCost(valueList)
            ascii_img[j] += gsval
    return ascii_img

cap = cv2.VideoCapture('Rick_Astley_Never_Gonna_Give_You_Up.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width, height)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
counter = 0
start = time.perf_counter()
last = start
f = open("temp_ascii_art.txt", 'w')

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        ascii_frame = cvt_ascii(frame)
        for row in ascii_frame:
            f.write(row + '\n')

        print(f"frame {counter} out of {frames}, {time.perf_counter() - last} seconds taken")
        last = time.perf_counter()
        counter+=1
        f.write('\n')
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
f.close()