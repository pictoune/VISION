import numpy as np
import cv2
from config import VIDEO_PATHS

from utils import ROI


def process_video(path):
	"""
	Processes a video by tracking an object within a defined region of interest (ROI).

	Args:
		path: The path to the video file.
	"""
	cap = cv2.VideoCapture(path)

	# Check if the video capture object was successfully created
	if not cap.isOpened():
		print(f"Error: Could not open video file {path}.")
		return

	try:
		# take first frame of the video
		ret,frame = cap.read()

		# load the image, clone it, and setup the mouse callback function
		clone = frame.copy()
		cv2.namedWindow("First image")
		myROI = ROI()
		cv2.setMouseCallback("First image", myROI.define_ROI)

		# keep looping until the 'q' key is pressed
		while True:
			# display the image and wait for a keypress
			cv2.imshow("First image", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the ROI is defined, draw it!
			if (myROI.roi_defined):
				# draw a green rectangle around the region of interest
				cv2.rectangle(frame, (myROI.r,myROI.c), (myROI.r+myROI.h,myROI.c+myROI.w), (0, 255, 0), 2)
			# else reset the image...
			else:
				frame = clone.copy()
			# if the 'q' key is pressed, break from the loop
			if key == ord("q"):
				break

		track_window = (myROI.r,myROI.c,myROI.h,myROI.w)

		# set up the ROI for tracking
		roi = frame[myROI.c:myROI.c+myROI.w, myROI.r:myROI.r+myROI.h]

		# conversion to Hue-Saturation-Value space
		# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

		# computation mask of the histogram:
		# Pixels with S<30, V<20 or V>235 are ignored 
		mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))

		# Marginal histogram of the Hue component
		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

		# Histogram values are normalised to [0,255]
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

		# Setup the termination criteria: either 10 iterations,
		# or move by less than 1 pixel
		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

		cpt = 1
		while True:
			ret ,frame = cap.read()
			if ret == True:
				hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				# Backproject the model histogram roi_hist onto the 
				# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
				dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

				# apply meanshift to dst to get the new location
				ret, track_window = cv2.meanShift(dst, track_window, term_crit)

				# Draw a blue rectangle on the current image
				r,c,h,w = track_window
				frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
				cv2.imshow('Sequence',frame_tracked)

				k = cv2.waitKey(60) & 0xff
				if k == 27:
					break
				elif k == ord('s'):
					cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
				cpt += 1
			else:
				break
		
	except KeyboardInterrupt:
		print("Interrupted by user, releasing resources.")
	
	finally:
		cap.release()
		cv2.destroyAllWindows()
		

if __name__ == "__main__":
	process_video(VIDEO_PATHS[0])