import os
import urllib.request
import cv2

MODEL_NAME = "object_tracking_vittrack_2023sep_int8bq.onnx"
MODEL_URL = f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_tracking_vittrack/{MODEL_NAME}"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_NAME)


def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_NAME} (~265 KB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def create_tracker(frame, bbox):
    params = cv2.TrackerVit_Params()
    params.net = MODEL_PATH
    tracker = cv2.TrackerVit_create(params)
    tracker.init(frame, bbox)
    return tracker


def main():
    download_model_if_missing()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera. Check camera permissions.")
        return

    win_name = "VitTrack (Lightweight Edge Tracker)"
    cv2.namedWindow(win_name)

    tracker = None

    print("Controls:")
    print("  - Press SPACE to freeze frame and draw bounding box")
    print("  - Press 'c' to clear tracker")
    print("  - Press 'q' or ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if tracker is None:
            cv2.putText(
                frame,
                "Position object, then press SPACE to select",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                bbox = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
                if bbox[2] > 0 and bbox[3] > 0:
                    tracker = create_tracker(frame, bbox)
            continue

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "VitTrack Tracking",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "TARGET LOST (press SPACE to reselect)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            "SPACE: Reselect | c: Clear | q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            tracker = None
        elif key == ord(" "):
            bbox = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
            if bbox[2] > 0 and bbox[3] > 0:
                tracker = create_tracker(frame, bbox)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
