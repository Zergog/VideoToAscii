import cv2
import numpy as np
import moviepy.editor as mpe

cap = cv2.VideoCapture('Final_Output_200Color.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width, height)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('inverted_silent.avi', fourcc, fps, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        inverted = gray = cv2.bitwise_not(frame)
        cv2.imshow('frame', inverted)
        out.write(inverted)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
my_clip = mpe.VideoFileClip('inverted_silent.avi')
audio = mpe.AudioFileClip('Rick_Astley_Never_Gonna_Give_You_Up.mp4')
final_clip = my_clip.set_audio(audio)
final_clip.write_videofile("Final_Output_200BNW.mp4")
cv2.destroyAllWindows()