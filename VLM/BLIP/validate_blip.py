import cv2
import numpy as np
from blip2itm import BLIP2ITMClient

itmclient = BLIP2ITMClient(port=12182)

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    cosine = itmclient.cosine(rgb_image, txt)
    return cosine

if __name__ == "__main__":
    image_filename = "R.jpg"
    label = "white flower"
    print("running main")
    bgr_image = cv2.imread(image_filename)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(get_itm_message(rgb_image, label))