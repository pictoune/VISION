import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cv2

def get_angle(vec):    
    i,j = vec
    angle = 0
    if(j == 0):
        if(i < 0):
            angle = np.pi/2
        elif(i > 0):
            angle = -np.pi/2
    elif(i == 0):
        if(j < 0):
            angle = np.pi
        elif(j > 0):
            angle = 0
    elif(i == 0 and j == 0):
        angle = 0
    elif(i < 0 and j > 0):
        angle = np.arctan(-i/j)
    elif(i < 0 and j < 0):
        angle = np.pi - np.arctan(i/j)
    elif(i > 0 and j < 0):
        angle = np.pi + np.arctan(-i/j)
    else:
        angle = -np.arctan(i/j)
        
    return np.rad2deg(angle)

def compute_grad(I):

    hy=np.array([[1/2,1,1/2]])
    hx=np.array([[-1/2,0,1/2]])
    Ix = signal.convolve2d(I, hy.T@hx)
    Iy = signal.convolve2d(I, hx.T@hy)

    mod = np.sqrt(np.square(Ix)+np.square(Iy))
    ori = np.zeros(mod.shape)
    for i in range(mod.shape[0]):
        for j in range(mod.shape[1]):
            ori[i,j] = get_angle((Ix[i,j],Iy[i,j]))

    return mod, ori

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

# cap = cv2.VideoCapture('./VOT-Ball.mp4')
# cap = cv2.VideoCapture('./Antoine_Mug.mp4')
# cap = cv2.VideoCapture('./VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('./VOT-Basket.mp4')
# cap = cv2.VideoCapture('./VOT-Car.mp4')
cap = cv2.VideoCapture('./VOT-Woman.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.namedWindow("orientations")
cv2.namedWindow("Hough transform")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
Rtable = {}
for k in range(21):
    Rtable[k]=[]
# set up the ROI for tracking
roi = frame[r:r+h,c:c+w]
center_x = c+w/2
center_y=r+h/2
clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
mod,ori = compute_grad(clone)
ori+=90
discret = 360/20
for i in range(roi.shape[0]):
    for j in range(roi.shape[1]):
        if mod[r+i,c+j]>20:
            Rtable[int(ori[r+i,c+j]/discret)].append((i-center_y+r,j-center_x+c))

while(1):
    key = cv2.waitKey(1)
    ret, frame = cap.read()
    if ret==True:
        # print('center',center_x,center_y)
        clone = frame.copy()
        clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        mod,ori = compute_grad(clone)
        ori+=90
        H = np.zeros(ori.shape)
        for l in range(ori.shape[0]):
            for m in range(ori.shape[1]):
                if mod[l,m] > 20:
                    for ind in Rtable[int(ori[l,m]/discret)]:
                        if l+ind[0]>= 0 and l+ind[0]<H.shape[0] and m+ind[1]>=0 and m+ind[1]<H.shape[1]:
                            H[int(l+ind[0]),int(m+ind[1])]+=1
        cv2.imshow("Hough transform", H/H.max())
        center_x, center_y = np.unravel_index(np.argmax(H), H.shape)
        # print(int(center_y-h/2),int(center_x-w/2),int(center_y+h/2),int(center_x+w/2))
        cv2.rectangle(frame, (int(center_y-h/2),int(center_x-w/2)), (int(center_y+h/2),int(center_x+w/2)), (0, 255, 0), 2)
        cv2.imshow("First image", frame)

        #display valid orientations
        ori = cv2.cvtColor(ori.astype('uint8'), cv2.COLOR_GRAY2BGR)
        ori[np.where(mod<20)] = [0,0,255]
        cv2.imshow("orientations", ori)
    else:
        break

cv2.destroyAllWindows()
cap.release()