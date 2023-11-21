import numpy as np
import cv2
from config import VIDEO_PATHS

from utils import ROI, compute_grad, process_roi


def process_video(path):
    """
    Processes a video by tracking an object within a defined region of interest (ROI).

    Args:
        path: The path to the video file.
    """
    cap = cv2.VideoCapture(path)

    # check if the video capture object was successfully created
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}.")
        return

    try:
        # take first frame of the video
        ret, frame = cap.read()

        # load the image, clone it, and setup the mouse callback function
        clone = frame.copy()
        cv2.namedWindow("First image")
        cv2.namedWindow("orientations")
        cv2.namedWindow("Hough transform")
        myROI = ROI()
        cv2.setMouseCallback("First image", myROI.define_ROI)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("First image", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the ROI is defined, draw it!
            frame = process_roi(frame, myROI, clone)
            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break

        track_window = (myROI.r, myROI.c, myROI.h, myROI.w)
        Rtable = {k: [] for k in range(21)}

        # set up the ROI for tracking
        roi = frame[myROI.r : myROI.r + myROI.h, myROI.c : myROI.c + myROI.w]
        center_x = myROI.c + myROI.w / 2
        center_y = myROI.r + myROI.h / 2
        clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        mod, ori = compute_grad(clone)
        ori += 90
        discret = 360 / 20

        # create indices for the entire ROI
        roi_indices = np.indices(roi.shape)
        i_indices, j_indices = roi_indices[0], roi_indices[1]

        # offset indices to match the position in the original image
        i_indices += myROI.r
        j_indices += myROI.c

        # flatten the indices for vectorized operations
        i_flat = i_indices.flatten()
        j_flat = j_indices.flatten()

        # apply the condition mod > 20
        condition = mod[i_flat, j_flat] > 20

        # filter indices and orientations based on the condition
        filtered_i = i_flat[condition]
        filtered_j = j_flat[condition]
        filtered_ori = ori[filtered_i, filtered_j]

        # compute displacements and update Rtable
        displacements = np.stack([filtered_i - center_y, filtered_j - center_x], axis=-1)
        for displacement, angle in zip(displacements, filtered_ori):
            Rtable[int(angle / discret)].append(tuple(displacement))

        # set up the ROI for tracking
        roi = frame[myROI.c : myROI.c + myROI.w, myROI.r : myROI.r + myROI.h]
        
        # conversion to Hue-Saturation-Value space
        # 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # computation mask of the histogram:
        # Pixels with S<30, V<20 or V>235 are ignored
        mask = cv2.inRange(
            hsv_roi, np.array((0.0, 30.0, 20.0)), np.array((180.0, 255.0, 235.0))
        )
        # marginal histogram of the Hue component
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        
        # histogram values are normalised to [0,255]
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # setup the termination criteria: either 10 iterations,
        # or move by less than 1 pixel
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while True:
            key = cv2.waitKey(1)
            ret, frame = cap.read()
            if not ret:
                break
            clone = frame.copy()
            clone = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
            mod, ori = compute_grad(clone[:, :, 2])
            ori += 90
            H = np.zeros(ori.shape)

            for l in range(ori.shape[0]):
                for m in range(ori.shape[1]):
                    if mod[l, m] > 20:
                        for ind in Rtable[int(ori[l, m] / discret)]:
                            if (
                                l + ind[0] >= 0
                                and l + ind[0] < H.shape[0]
                                and m + ind[1] >= 0
                                and m + ind[1] < H.shape[1]
                            ):
                                H[int(l + ind[0]), int(m + ind[1])] += 1

            cv2.imshow("Hough transform", H / H.max())

            ret, track_window = cv2.meanShift(H, track_window, term_crit)

            # draw a blue rectangle on the current image
            r, c, h, w = track_window
            frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
            cv2.imshow("First image", frame_tracked)

            # display valid orientations
            ori = cv2.cvtColor(ori.astype("uint8"), cv2.COLOR_GRAY2BGR)
            ori[np.where(mod < 20)] = [0, 0, 255]
            cv2.imshow("orientations", ori)
    
    except KeyboardInterrupt:
        print("Interrupted by user, releasing resources.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video(VIDEO_PATHS[0])
