import argparse
import requests
import sounddevice as sd
import numpy as np
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100 
CHANNELS = 1
OUTPUT_FILENAME = 'temp_recording.wav'

def record_audio():
    recorded_frames = []

    def callback(indata, frames, time, status):
        if status:
            logger.warning(status)
        recorded_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
    with stream:
        logger.info('Recording started... Press Ctrl+C to stop.')
        try:
            while True:
                sd.sleep(1000)  # Sleep in intervals to allow keyboard interrupt
        except KeyboardInterrupt:
            logger.info('Recording stopped by user.')

    if not recorded_frames:
        logger.error('No audio was recorded.')
        return None

    recording = np.concatenate(recorded_frames, axis=0)
    sf.write(OUTPUT_FILENAME, recording, SAMPLE_RATE)
    logger.info(f'Audio saved to {OUTPUT_FILENAME}')
    return OUTPUT_FILENAME

def send_to_api(audio_path, api_url):
    if not audio_path:
        return

    logger.info(f'Sending {audio_path} to API at {api_url}...')
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path, f, 'audio/wav')}
            data = {'prompt': '将音频转录为文字，并在文本最后附加<情感>标签，标签类型涵盖：sad，angry，neutral，happy，surprise，fear，disgust，还有other。'}
            
            response = requests.post(api_url, files=files, data=data)
            response.raise_for_status()

        result = response.json()

        print('\n' + '='*20)
        print(' Transcription Result')
        print('='*20)
        if 'transcription' in result:
            print(result['transcription'])
        elif 'error' in result:
            print(f'An error occurred on the server: {result['error']}')
        else:
            print(f'Received an unexpected response: {result}')
        print('='*20 + '\n')

    except requests.exceptions.RequestException as e:
        logger.error(f'Failed to connect to the API server: {e}')
        print('\nError: Could not get a response from the server.')
        print(f'Please ensure the API server is running at {api_url}\n')


if __name__ == '__main__':
    audio_file = record_audio()
    api_url = 'http://127.0.0.1:6543/transcribe'
    send_to_api(audio_file, api_url)
