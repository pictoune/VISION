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
            frame = process_roi(frame, myROI, clone)
            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break

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

        # filter indices based on the condition
        filtered_indices = mod[i_indices, j_indices] > 20
        i_filtered = i_flat[filtered_indices]
        j_filtered = j_flat[filtered_indices]

        # compute the orientation and the displacement
        orientations = ori[i_filtered, j_filtered]
        displacements = np.stack([(i_filtered - center_y, j_filtered - center_x)], axis=-1)

        # map the filtered values to Rtable
        for orient, disp in zip(orientations, displacements):
            Rtable[int(orient / discret)].append(tuple(disp))

        while True:
            key = cv2.waitKey(1)
            ret, frame = cap.read()

            if not ret:
                break

            clone = frame.copy()
            clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
            mod, ori = compute_grad(clone)
            ori += 90
            H = np.zeros(ori.shape)

            for l in range(ori.shape[0]):
                for m in range(ori.shape[1]):
                    if mod[l, m] > 20:
                        for ind in Rtable[int(ori[l, m] / discret)]:
                            if (
                                0 <= l + ind[0] < H.shape[0]
                                and 0 <= m + ind[1] < H.shape[1]
                            ):
                                H[int(l + ind[0]), int(m + ind[1])] += 1

            cv2.imshow("Hough transform", H / H.max())
            center_x, center_y = np.unravel_index(np.argmax(H), H.shape)
            cv2.rectangle(
                frame,
                (int(center_y - myROI.h / 2), int(center_x - myROI.w / 2)),
                (int(center_y + myROI.h / 2), int(center_x + myROI.w / 2)),
                (0, 255, 0),
                2,
            )
            cv2.imshow("First image", frame)

            ori = cv2.cvtColor(ori.astype("uint8"), cv2.COLOR_GRAY2BGR)
            ori[np.where(mod < 20)] = [0, 0, 255]
            cv2.imshow("orientations", ori)
    
    except KeyboardInterrupt:
        print("Interrupted by user, releasing resources.")
    
    finally:
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    process_video(VIDEO_PATHS[0])
