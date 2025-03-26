# Facial-Emotion-Recognizer

Facial-Emotion-Recognizer is a simple Python tool that processes video files to detect and classify facial emotions. It leverages OpenCV for video processing and face detection, and uses a Keras model (VGG16) for emotion recognition, which was pre-trained on an undersampled version of the FER2013 dataset. The model achieves a training accuracy of 94.81%, and a validation accuracy set of 84.60%. 

The script outputs a JSON file with timestamped emotion predictions in 7 categories: `anger, disgust, fear, enjoyment, contempt, sadness, surprise`.


## Requirements

The project depends on the following Python packages:

- [opencv-python==4.7.0.72](https://pypi.org/project/opencv-python/)
- [numpy==1.24.2](https://pypi.org/project/numpy/)
- [keras==2.11.0](https://pypi.org/project/Keras/)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the script by passing the video file as an argument:

```bash
python facial_emotions.py <video_filename>
```

For example:

```bash
python facial_emotions.py sample_video.mp4
```

The script will process the video, detect faces, classify emotions, and save the results in a JSON file (named after the video file, e.g., `sample_video.json`).


## More Information

Check the [original repo](https://github.com/alex-pt01/The-Role-of-Facial-Emotions-in-Usability-Evaluation/) by Alexandre Rodrigues.


## Citation

```bibtex
@article{rodrigues2025emotionality,
  title={The Emotionality Tool: Evaluating Usability with Facial Emotions Analysis},
  author={Rodrigues, Alexandre Antunes and Santos, Flávia de Souza and Gama, Sandra Pereira},
  journal={International Journal of Human-Computer Interaction},
  year={2025},
  publisher={Taylor \& Francis},
  note={In press},
}
```

