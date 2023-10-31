import os
import cv2
import numpy as np

video_path = '05300535.mp4'

def extract_frames(video_path, start_frame, end_frame):
    # Extract frames from the video and save them to the 'frames' directory
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count <= end_frame:
        ret, frame = cap.read()
        if ret and frame_count >= start_frame:
            small_frame = cv2.resize(frame, (256, 256))
            frames.append(small_frame)
        frame_count += 1
    cap.release()

    # Create a directory to save the frames if it does not exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # Save the frames to the directory
    video_name = os.path.basename(video_path).split('.')[0]
    for i, frame in enumerate(frames):
        save_path = f'frames/{video_name}_{start_frame + i}.jpg'
        cv2.imwrite(save_path, frame)
        print(f"Frame saved at {save_path}")

    return frames
