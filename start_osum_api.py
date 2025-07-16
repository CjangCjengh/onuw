import argparse
import logging
import time
import os
import tempfile

import torch
import torchaudio
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Speech Transcription API', version='1.0')

app.state.model_objects = {}

@app.on_event('startup')
def load_model_on_startup():
    logger.info('Starting server and loading model...')
    config_path = 'examples/osum/conf/config_llm_huawei_base-version.yaml'
    checkpoint_path = 'OSUM/infer.pt'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint file not found at {checkpoint_path}. Please update the path.')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found at {config_path}. Please update the path.')

    args = GxlNode({'checkpoint': checkpoint_path})
    configs = utils_file.load_dict_from_yaml(config_path)
    
    model, configs = init_model(args, configs)

    if torch.cuda.is_available():
        gpu_id = 0
        model = model.cuda(gpu_id)
        logger.info(f'Model loaded on GPU: {gpu_id}')
    else:
        gpu_id = -1
        logger.warning('CUDA not available. Running model on CPU.')

    tokenizer = init_tokenizer(configs)

    app.state.model_objects = {
        'model': model,
        'tokenizer': tokenizer,
        'gpu_id': gpu_id,
        'resample_rate': 16000
    }
    logger.info('Model and tokenizer loaded successfully.')

def do_resample(input_wav_path, output_wav_path, resample_rate):
    logger.debug(f'Resampling {input_wav_path} to {output_wav_path}')
    waveform, sample_rate = torchaudio.load(input_wav_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
    resampled_waveform = resampler(waveform)
    
    utils_file.makedir_for_file(output_wav_path)
    torchaudio.save(output_wav_path, resampled_waveform, resample_rate)

def do_decode(input_wav_path: str, input_prompt: str):
    model_data = app.state.model_objects
    model = model_data['model']
    gpu_id = model_data['gpu_id']
    resample_rate = model_data['resample_rate']

    logger.info(f'Processing request with prompt: {input_prompt}')

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_resampled_file:
        resampled_path = tmp_resampled_file.name

    try:
        do_resample(input_wav_path, resampled_path, resample_rate)

        waveform, sample_rate = torchaudio.load(resampled_path)
        waveform = waveform.squeeze(0)

        window = torch.hann_window(400)
        stft = torch.stft(waveform, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=400, n_mels=80))
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feat = log_spec.transpose(0, 1)

        device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        feat = feat.unsqueeze(0).to(device)
        feat_lens = torch.tensor([feat.shape[1]], dtype=torch.int64).to(device)

        res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
        logger.info(f'Result: {res_text}')
        return res_text

    finally:
        if os.path.exists(resampled_path):
            os.remove(resampled_path)

@app.post('/transcribe')
async def transcribe_audio(
    prompt: str = Form(''),
    file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_audio_file:
        tmp_audio_file.write(await file.read())
        tmp_audio_path = tmp_audio_file.name

    try:
        transcription = do_decode(tmp_audio_path, prompt)
        return {'transcription': transcription}
    except Exception as e:
        logger.error(f'An error occurred during transcription: {e}')
        return {'error': str(e)}
    finally:
        os.remove(tmp_audio_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6543)
    args = parser.parse_args()

    uvicorn.run(app, host='0.0.0.0', port=args.port)
