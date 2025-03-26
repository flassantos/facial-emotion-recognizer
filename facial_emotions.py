import cv2
import numpy as np
from keras.models import load_model
import json
import sys

def format_video_time(frame_no, fps):
    """Format the current frame number and fps into a timestamp string."""
    frame_time = frame_no / fps
    total_seconds = int(frame_time)
    milliseconds = int((frame_time - total_seconds) * 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    # Include hours in the timestamp only if needed
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

def get_emotions_dict(video_path):
    """Process the video and return a dictionary mapping timestamps to emotion predictions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: Failed to open video file')
        return {}

    # Load the emotion detection model
    try:
        emotion_model = load_model('model.h5')
    except Exception as e:
        print("Error loading model:", e)
        cap.release()
        return {}

    # Load the Haar cascade for face detection once
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    time_emotions = {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # Fallback FPS if retrieval fails

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to a fixed resolution
        frame = cv2.resize(frame, (1280, 720))
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = format_video_time(frame_no, fps)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        predictions = []
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            try:
                face_roi_resized = cv2.resize(face_roi, (48, 48))
            except Exception as e:
                print(f"Error resizing face region: {e}")
                continue
            # Prepare the face region for prediction: (1, 48, 48, 1)
            face_roi_expanded = np.expand_dims(np.expand_dims(face_roi_resized, axis=0), axis=-1)
            emotion_prediction = emotion_model.predict(face_roi_expanded)
            predictions.append(emotion_prediction[0].tolist())

        # Only store predictions if at least one face is detected
        if predictions:
            time_emotions[timestamp] = predictions

        # Optional: Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return time_emotions

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_filename>")
        sys.exit(1)
    video_filename = sys.argv[1]
    emotions = get_emotions_dict(video_filename)
    output_filename = video_filename.rsplit('.', 1)[0] + ".json"
    with open(output_filename, 'w') as json_file:
        json.dump(emotions, json_file, indent=4)
    print(f"Emotion predictions saved to {output_filename}")

if __name__ == "__main__":
    main()
