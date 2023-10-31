import os
import cv2
import numpy as np

video_path = '05300535.mp4'

def extract_frames(video_path, target_frames):
    # Extract frames from the video and save them to the 'frames' directory
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < target_frames:
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, (256, 256))
            frames.append(small_frame)
            frame_count += 1
        else:
            break
    cap.release()
    while len(frames) < target_frames:
        if len(frames) > 0:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((256, 256, 3), dtype=np.uint8))

    # Create a directory to save the frames if it does not exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # Save the frames to the directory
    video_name = os.path.basename(video_path).split('.')[0]
    for i, frame in enumerate(frames):
        save_path = f'frames/{video_name}_{i}.jpg'
        cv2.imwrite(save_path, frame)
        print(f"Frame saved at {save_path}")

    return frames

extract_frames(video_path, 16)  # Change the value of target_frames as needed
