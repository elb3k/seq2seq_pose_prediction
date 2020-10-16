import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

def draw(img, keypoints, color=(255, 0, 0) ):

    img = np.copy(img)
    for c in keypoints:
        img = cv2.circle(img, ( int(c[0]), int(c[1]) ), 5, color, -1)
    
    return img



sample = pd.read_pickle("sample_v2.pkl")
data = sample["data"]
output = sample["output"]

for i in tqdm(range(len(output)), desc="Preprocess"):
    x, y, p = output[i]
    
    x = x + data[74+i]
    y = y + data[74+i]
    p = p + data[74+i]

    output[i] = (x, y, p)

video = cv2.VideoCapture("scan_video.avi")
images = []
status, img = video.read()

print("[INFO] Reading video")

while status:
    images.append(img)
    status, img = video.read()

video.release()


for i in tqdm(range(len(output)), desc="Making Videos"):
    imgs = images[i: i+150]
    src = imgs[:75]
    target = imgs[75:]

    x, y, p = output[i]

    out = cv2.VideoWriter("vis/%d.mp4"%(i+1), cv2.VideoWriter_fourcc(*"MP4V"), 25.0, (1920, 1080))
    
    for j in range(len(x)):
        img = draw(src[j], x[j])
        out.write(img)

    for j in range(len(y)):
        img = draw(target[j], y[j], color=(0, 255, 0))
        img = draw(img, p[j], color=(0, 0, 255))
        out.write(img)
    
    out.release()
