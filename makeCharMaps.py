import string
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd

charDimension = 40
H, W = charDimension, charDimension
def imgArrayToGrayDict(mask):
    grayDict = {}
    index = 0
    for row in mask:
        for pixel in row:
            value = pixel[0]
            grayDict[index] = value
            index += 1
    return grayDict
def charMap(char):
    ascii_img = []
    mask = np.zeros((H, W, 3), np.uint8)
    pil_mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_mask)
    font = ImageFont.truetype("COURIER.TTF", int(H))
    draw.text((0, 0), char, font=font, fill=(255, 255, 255))
    pil_mask = np.array(pil_mask)
    return pil_mask

print(charMap("a"))
print(imgArrayToGrayDict(charMap("a")))
chars = string.printable
chars = chars.strip() + " "
df = pd.DataFrame(index=[char for char in chars], columns = range(H*W))
for char in chars:
    df.loc[char] = imgArrayToGrayDict(charMap(char))
df = df.mod(255)
print(list(df.loc["a"]))
df.to_csv("grayChars.csv")