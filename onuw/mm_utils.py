import cv2
import pyaudio
import wave
import threading
import os
import re
import requests
from moviepy.editor import VideoFileClip, AudioFileClip


EMOTION_API_URL = 'http://127.0.0.1:5432/analyze_emotion'
TRANSCRIPTION_API_URL = 'http://127.0.0.1:6543/transcribe'
TEMP_AUDIO_FILE = 'temp_audio_for_analysis.wav'
TEMP_VIDEO_FILE = 'temp_video_for_analysis.avi'
FINAL_VIDEO_FILE = 'final_recording_for_analysis.mp4'
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK_SIZE = 1024
EMOTIONS = ['sad', 'anger', 'neutral', 'happy', 'surprise', 'fear', 'disgust', 'other']

stop_event = threading.Event()

def _record_audio(output_filename: str):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)
    except Exception:
        stop_event.set()
        return

    frames = []
    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK_SIZE)
            frames.append(data)
        except IOError:
            pass

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def _record_video(output_filename: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        stop_event.set()
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            cv2.imshow('Recording... (Press Q to stop)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1)

def _combine_audio_video(video_path: str, audio_path: str, output_path: str) -> bool:
    try:
        if not all(os.path.exists(p) for p in [video_path, audio_path]): return False
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        video_clip.close()
        audio_clip.close()
        return True
    except Exception:
        return False

def _call_emotion_api(video_path: str, results: dict):
    if not os.path.exists(video_path):
        results['face'] = 'other'
        return
    
    files = {'video': (os.path.basename(video_path), open(video_path, 'rb'), 'video/mp4')}
    payload = {'question': 'Please determine which emotion label in the video represents: happy, sad, neutral, angry, surprise, disgust, fear, other.'}
    try:
        response = requests.post(EMOTION_API_URL, files=files, data=payload, timeout=300)
        if response.status_code == 200:
            emotion_label = response.json()['emotion_label'].strip()
            if emotion_label not in EMOTIONS:
                emotion_label = 'other'
            results['face'] = emotion_label
        else:
            results['face'] = 'other'
    except requests.exceptions.RequestException as e:
        results['face'] = 'other'

def _call_transcription_api(audio_path: str, results: dict):
    if not os.path.exists(audio_path):
        results['speech'] = ''
        results['tone'] = 'other'
        return

    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            data = {'prompt': '将音频转录为文字，并在文本最后附加<情感>标签，标签类型涵盖：sad，angry，neutral，happy，surprise，fear，disgust，还有other。'}
            response = requests.post(TRANSCRIPTION_API_URL, files=files, data=data, timeout=300)
            if response.status_code == 200:
                speech = response.json()['transcription'].strip()
                tone_label = 'other'
                emo_match = re.search(r'^(.*)<(.*)>', speech)
                if emo_match:
                    speech = emo_match.group(1)
                    tone_label = emo_match.group(2)
                if tone_label not in EMOTIONS:
                    tone_label = 'other'
                results['speech'] = speech
                results['tone'] = tone_label
            else:
                results['speech'] = ''
                results['tone'] = 'other'
    except requests.exceptions.RequestException as e:
        results['speech'] = ''
        results['tone'] = 'other'

def record_and_analyze() -> dict:
    print('Recording... Press Q to stop.')
    global stop_event
    stop_event = threading.Event()
    results = {'thought': ''}

    audio_thread = threading.Thread(target=_record_audio, args=(TEMP_AUDIO_FILE,))
    audio_thread.start()

    try:
        _record_video(TEMP_VIDEO_FILE)
    except KeyboardInterrupt:
        pass
    finally:
        if audio_thread.is_alive():
            stop_event.set()
            audio_thread.join()

    if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [TEMP_AUDIO_FILE, TEMP_VIDEO_FILE]):
        for f in [TEMP_AUDIO_FILE, TEMP_VIDEO_FILE]:
            if os.path.exists(f): os.remove(f)
        return {'thought': '', 'speech': '', 'face': 'other', 'tone': 'other'}

    if not _combine_audio_video(TEMP_VIDEO_FILE, TEMP_AUDIO_FILE, FINAL_VIDEO_FILE):
        for f in [TEMP_AUDIO_FILE, TEMP_VIDEO_FILE]:
            if os.path.exists(f): os.remove(f)
        return {'thought': '', 'speech': '', 'face': 'other', 'tone': 'other'}

    emotion_thread = threading.Thread(target=_call_emotion_api, args=(FINAL_VIDEO_FILE, results))
    transcription_thread = threading.Thread(target=_call_transcription_api, args=(TEMP_AUDIO_FILE, results))

    emotion_thread.start()
    transcription_thread.start()

    emotion_thread.join()
    transcription_thread.join()

    for f in [TEMP_AUDIO_FILE, TEMP_VIDEO_FILE, FINAL_VIDEO_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    return results

if __name__ == '__main__':
    analysis_results = record_and_analyze()
    print(analysis_results)
