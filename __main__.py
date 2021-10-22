import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(0)
hsv_min= (29, 86, 6)
hsv_max= (64, 255, 255)

def find_object(frame, mascara, color):
	cnts, hierarchy = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key= cv2.contourArea)

	x,y,w,h = cv2.boundingRect(c)
	cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
	return (round(x+w/2), round(y+h/2))

while True:
	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mascara = cv2.inRange(hsv, hsv_min, hsv_max)
	pg= find_object(frame, mascara, (255,0,0))

	cv2.imshow('frame', frame)
	#cv2.imshow('camara' ,mascara)

	k = cv2.waitKey(1) & 0xFF
	if k==27:
		break 