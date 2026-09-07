import time
import cv2
import numpy as np
import torch
from transformers import EdgeTamVideoModel, Sam2VideoProcessor

MODEL_ID = "yonigozlan/EdgeTAM-hf"

# Global click state
click_point = None


def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


def get_device_and_dtype():
    """Auto-detect best available hardware accelerator."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def main():
    global click_point

    device, dtype = get_device_and_dtype()

    print(f"Loading EdgeTAM ({MODEL_ID}) on {device.upper()} ({dtype})...")
    model = EdgeTamVideoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)
    session = processor.init_video_session(inference_device=device, dtype=dtype)
    print("EdgeTAM model ready!\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not access camera. Check OS Camera privacy permissions.")
        return

    win_name = "EdgeTAM Live Tracker"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    tracking = False
    prev_time = time.time()

    print("Controls:")
    print("  - Left click on any object to segment & track it immediately")
    print("  - Press SPACE to freeze frame and draw a bounding box")
    print("  - Press 'c' to clear tracker")
    print("  - Press 'q' or ESC to quit\n")

    while True:
        # Grab latest frame to avoid buffer latency
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break

        # Mirror video feed
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Handle point-click prompt
        if click_point is not None:
            cx, cy = click_point
            click_point = None

            session.reset_inference_session()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, device=device, return_tensors="pt")
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=1,
                input_points=[[[[cx, cy]]]],
                input_labels=[[[1]]],
                original_size=inputs.original_sizes[0],
            )
            tracking = True

        # Process tracking if active
        if tracking:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, device=device, return_tensors="pt")

            with torch.no_grad():
                out = model(inference_session=session, frame=inputs.pixel_values[0].to(dtype))

            masks = processor.post_process_masks(
                [out.pred_masks], original_sizes=inputs.original_sizes, binarize=True
            )[0]
            mask = masks[0, 0].cpu().numpy().astype(np.uint8)

            ys, xs = np.where(mask > 0)
            if len(xs) > 30:
                # Translucent mask overlay
                color_layer = np.zeros_like(frame)
                color_layer[mask > 0] = (0, 255, 120)  # Bright green
                cv2.addWeighted(color_layer, 0.45, frame, 1.0, 0, frame)

                # Bounding box
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 120), 2)
                cv2.putText(
                    frame,
                    "EdgeTAM Tracking",
                    (x_min, max(20, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 120),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "TARGET LOST (click or press SPACE)",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "Click on an object OR press SPACE to draw box",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        # FPS & HUD
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Device: {device.upper()}",
            (w - 240, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            "Click: Point Track | SPACE: Box Select | c: Clear | q: Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("c"):
            session.reset_inference_session()
            tracking = False
        elif key == ord(" "):
            # Freeze frame and let user draw bounding box
            roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
            if roi[2] > 10 and roi[3] > 10:
                session.reset_inference_session()
                rx, ry, rw, rh = roi
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs = processor(images=rgb, device=device, return_tensors="pt")
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=1,
                    input_boxes=[[[rx, ry, rx + rw, ry + rh]]],
                    original_size=inputs.original_sizes[0],
                )
                tracking = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
