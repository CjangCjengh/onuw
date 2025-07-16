import os
import argparse
import uvicorn
import torch
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

chat_model = None
device = None
vis_processor = None

app = FastAPI()

class SimpleArgs:
    def __init__(self, cfg_path, options=None):
        self.cfg_path = cfg_path
        self.options = options if options is not None else []

def load_model_and_processor():
    print('Loading...')
    args = SimpleArgs(cfg_path='eval_configs/demo.yaml')
    cfg = Config(args)

    model_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(model_device)

    vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    model.eval()
    chat = Chat(model, processor, device=model_device)
    print('Loaded!')

    global chat_model, device, vis_processor
    chat_model = chat
    device = model_device
    vis_processor = processor

def process_video_question(video_path, question):
    if chat_model is None:
        raise RuntimeError('Unloaded')

    chat_state = Conversation(
        system='',
        roles=(r'<s>[INST] ', r' [/INST]'),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep='',
    )

    img_list = []
    chat_state.append_message(chat_state.roles[0], '<video><VideoHere></video> <feature><FeatureHere></feature>')

    chat_model.ask(question, chat_state)

    img_list.append(video_path)
    chat_model.encode_img(img_list)

    response = chat_model.answer(
        conv=chat_state,
        img_list=img_list,
        temperature=0.2,
        max_new_tokens=500,
        max_length=2000
    )[0]
    
    return response

@app.on_event('startup')
async def startup_event():
    load_model_and_processor()

@app.post('/analyze_emotion')
async def analyze_emotion(
    video: UploadFile = File(...),
    question: str = Form('Please determine which emotion label in the video represents: happy, sad, neutral, angry, surprise, disgust, fear, other.')
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            content = await video.read()
            tmp_video.write(content)
            tmp_video_path = tmp_video.name
        
        print(f'Saved: {tmp_video_path}')
        print(f'Prompt: {question}')

        model_response = process_video_question(tmp_video_path, question)
        print(f'Response: {model_response}')

        return {'emotion_label': model_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error: {str(e)}')
    finally:
        if 'tmp_video_path' in locals() and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
            print(f'Deleted: {tmp_video_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5432)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
