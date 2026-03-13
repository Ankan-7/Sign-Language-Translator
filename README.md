# Sign Language Translator (Real-Time)

A real-time sign language translation system built using **Computer Vision and Machine Learning**.

The system detects hand gestures using MediaPipe and translates them into text in real time.

## Features

• Real-time hand gesture recognition  
• Machine learning based letter prediction  
• Word autocomplete suggestions using NLP  
• Gesture commands (SPACE, CLEAR, DELETE, SPEAK)  
• Text-to-speech output  
• Smooth prediction using buffer stabilization  

## Tech Stack

Python  
OpenCV  
MediaPipe  
Scikit-Learn  
WordFreq (NLP autocomplete)

## Project Structure

```
dataset/       → Gesture datasets  
src/           → Source code  
model/         → Trained ML model  
predict_sign.py → Main application  
```

## How It Works

1. MediaPipe detects hand landmarks.
2. Landmarks are fed into a trained ML model.
3. The model predicts the corresponding letter.
4. Letters form words and sentences.
5. NLP suggests possible word completions.
6. The system can speak the sentence aloud.

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the program:

```
python src/predict_sign.py
```

## Example Commands

SPACE → Separate words  
CLEAR → Clear sentence  
DELETE → Remove last letter  
SPEAK → Convert sentence to speech  

## Future Improvements

• Full A–Z gesture support  
• Deep learning based model  
• GUI interface  
• Higher prediction accuracy

## Demo

![Demo](demo.png)
