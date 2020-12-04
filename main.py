import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import moviepy.editor as mpe
import time

gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~i!lI;:,\"^`'. "
gscale2 = "@%#*+=-:. "
resolution = 6


def get_average_gray(arr):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    width, height = arr.shape
    return np.average(arr.reshape(width*height))
def get_average_chroma(arr):
    avg_color_per_row = np.average(arr, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
    int_averages = (int(avg_colors[2]), int(avg_colors[1]), int(avg_colors[0]))
    # print(int_averages)
    return int_averages

def cvt_ascii(img, cols, scale):
    global gscale1, gscale2, resolution
    H, W = img.shape[0]*resolution, img.shape[1]*resolution
    img = cv2.resize(img, (W, H))
    w = W / cols
    h = w / scale
    rows = int(H/h)
    ascii_img = []
    mask = np.zeros((H, W, 3), np.uint8)
    # print([element for element in mask.shape])
    pil_mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_mask)
    font = ImageFont.truetype("COURIER.TTF", int(h))
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
            avg = int(get_average_gray(cropped))
            avg_rgb = get_average_chroma(cropped)
            gsval = gscale1[int((avg*69)/255)]
            ascii_img[j] += gsval
            draw.text((x1, y1), gsval, font=font, fill=(255, 255, 255))
    f = open("temp_ascii_art.txt", 'w')
    for row in ascii_img:
        f.write(row + '\n')
    f.close()
    # encoded_image_path = text_to_image.encode_file("temp_ascii_art.txt", "output.png")
    # for line_num, line in enumerate(ascii_img):
    #     draw.text((0, int((line_num * h) + h/2)), line, font=font)
    mask = cv2.bitwise_not(cv2.cvtColor(np.array(pil_mask), cv2.COLOR_RGB2BGR))
    cv2.imwrite("frame.png", img)
    cv2.imwrite("mask.png", mask)
    # hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    # hsv_frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (W, H))
    # hsv_final = cv2.bitwise_and(hsv_frame, hsv_frame, mask= hsv_mask)
    return mask




cap = cv2.VideoCapture('Rick_Astley_Never_Gonna_Give_You_Up.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width, height)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width*resolution, height*resolution))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
counter = 0
start = time.perf_counter()
last = start

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        ascii_frame = cvt_ascii(frame, 200, 0.43)
        out.write(ascii_frame)

        print(f"frame {counter} out of {frames}, {time.perf_counter() - last} seconds taken")
        last = time.perf_counter()
        counter+=1
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
my_clip = mpe.VideoFileClip('output.avi')
audio = mpe.AudioFileClip('Rick_Astley_Never_Gonna_Give_You_Up.mp4')
final_clip = my_clip.set_audio(audio)
final_clip.write_videofile("Final_Output_200Color.mp4")
cv2.destroyAllWindows()
end = time.perf_counter()
print(f"Total Time Spent Converting: {end - start:0.4f} seconds.")