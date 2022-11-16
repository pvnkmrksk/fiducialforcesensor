import numpy as np
import cv2
import datetime
import cv2.aruco as aruco


def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
    )  # depends on fourcc available camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 120)
    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    fps_sleep = int(1000 / cap_fps)
    print("* Capture width:", cap_width)
    print("* Capture height:", cap_height)
    print("* Capture FPS:", cap_fps, "ideal wait time between frames:", fps_sleep, "ms")

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    print(type(aruco_dict))
    # print the aruco dictionary

    aruco_params = aruco.DetectorParameters_create()

    # main loop: retrieves and displays a frame from the camera
    while False:
        # blocks until the entire frame is read
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        frames += 1

        # compute fps: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)
        print("FPS:", cur_fps, "elapsed time:", elapsed_time, "frames:", frames)
        # draw FPS text and display image
        cv2.putText(
            img,
            "FPS: " + str(cur_fps),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("webcam", img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()

# But the fps is still 30. I tried to measure the fps of the live video by using a video file, and it works fine. I am not sure whether it is the problem of the camera or the way I measure the fps. I am using a USB camera, and the model is the same as the one in the link: https://www.amazon.com/gp/product/B07F2X8Q2K/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1

# I am using python 3.7.3 and opencv

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()
