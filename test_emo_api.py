import cv2
import pyaudio
import wave
import threading
import os
import requests
import argparse
import time
from moviepy.editor import VideoFileClip, AudioFileClip

stop_event = threading.Event()

def record_audio(output_filename: str, chunk_size=1024, audio_format=pyaudio.paInt16, channels=1, rate=44100):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    except Exception as e:
        print(f'Could not open microphone. Error: {e}')
        stop_event.set()
        return

    print('Starting audio recording...')
    frames = []

    while not stop_event.is_set():
        try:
            data = stream.read(chunk_size)
            frames.append(data)
        except IOError:
            pass

    print('Stopping audio recording...')
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f'Audio saved to {output_filename}')

def record_video(output_filename: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        stop_event.set()
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_video_filename = 'temp_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_filename, fourcc, fps, (width, height))

    print('Starting video recording... Press Ctrl+C in the terminal to stop.')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            # Display the recording window
            cv2.imshow('Recording... (Press Ctrl+C in terminal to stop)', frame)
            # Allow 'q' to be pressed in the window to quit, as an alternative to Ctrl+C
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print('Stopping video recording...')
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # Ensure all OpenCV windows are closed
        for i in range(5):
            cv2.waitKey(1)
        print(f'Video saved to {temp_video_filename}')


def record_and_analyze(api_url: str):
    '''Main function to coordinate recording, merging, and API requests.'''
    temp_audio_file = 'temp_audio.wav'
    temp_video_file = 'temp_video.avi'
    final_output_file = 'final_recording.mp4'

    # Start the audio recording in a separate thread
    audio_thread = threading.Thread(target=record_audio, args=(temp_audio_file,))
    audio_thread.start()

    try:
        record_video(temp_video_file)
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Preparing to stop all processes...')

    print('\nWaiting for the audio thread to finish...')
    stop_event.set()
    audio_thread.join()

    if not os.path.exists(temp_video_file) or not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
        print('Recording failed. Missing or empty temporary files.')
        for f in [temp_audio_file, temp_video_file]:
             if os.path.exists(f): os.remove(f)
        return

    if combine_audio_video(temp_video_file, temp_audio_file, final_output_file):
        send_to_api(api_url, final_output_file)

    print('\nCleaning up files...')
    for f in [temp_audio_file, temp_video_file, final_output_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f'   - Deleted {f}')

def combine_audio_video(video_path: str, audio_path: str, output_path: str):
    '''Combines video and audio files using moviepy.'''
    print('\nMerging audio and video...')
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        # logger=None hides the progress bar from moviepy
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        video_clip.close()
        audio_clip.close()
        print(f'Merge complete. Final video saved as {output_path}')
        return True
    except Exception as e:
        print(f'Merge failed: {e}')
        return False

def send_to_api(api_url: str, video_path: str):
    print('\nSending video to the server for analysis...')
    if not os.path.exists(video_path):
        print(f'Error: Video file {video_path} not found.')
        return

    files = {'video': (os.path.basename(video_path), open(video_path, 'rb'), 'video/mp4')}
    question = 'Please determine which emotion label in the video represents: happy, sad, neutral, angry, surprise, disgust, fear, other.'
    payload = {'question': question}

    try:
        start_time = time.time()
        response = requests.post(api_url, files=files, data=payload, timeout=300)
        end_time = time.time()
        print(f'⏱️ Server processing time: {end_time - start_time:.2f} seconds')

        if response.status_code == 200:
            result = response.json()
            print('\n================= Analysis Result =================')
            print(f'Emotion Label: {result.get('emotion_label', 'N/A')}')
            print('==========================================')
        else:
            print(f'\nRequest failed with status code: {response.status_code}')
            print(f'Error details: {response.text}')

    except requests.exceptions.RequestException as e:
        print(f'\nAn error occurred while connecting to the server: {e}')

if __name__ == '__main__':
    url = 'http://127.0.0.1:5432/analyze_emotion'
    record_and_analyze(api_url=url)
