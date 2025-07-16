### Installation

```sh
conda create -n onuw python=3.10
conda activate onuw
pip install -r requirements.txt
pip install -r requirements_mm.txt
```

### Setting APIs

Create `.env` file to set APIs.
```sh
vim .env
```

```sh
export OPENAI_API_KEY="Your API Key"
export OPENAI_API_BASE="Your API Base"
export OPENAI_API_KEY_EMB="Your API Key for Embedding Model"
export OPENAI_API_BASE_EMB="Your API Base for Embedding Model"
```

Change `model name` in `onuw/backends/openai.py`, Line 22:
```python
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
```

### Configuration
`configs/werewolf.json`

### Running Games
```sh
python main.py --env Werewolf --num_runs <number of runs of different settings> --num_repeats <number of repeating runs in one setting> --random --cli --save_path <save path for game logs>
```
The following options are only enabled when added:

- `random`: Randomly assign roles at the beginning
- `cli`: Launch cli (the interactive interface in the command line)

### Human Participation
#### Setting Emotion-LLaMA API
```sh
git clone https://github.com/ZebangCheng/Emotion-LLaMA
cd Emotion-LLaMA
conda env create -f environment.yml
```
Follow the instructions in [ZebangCheng/Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA) to install dependencies, download the model, and set the path in the config.
```sh
cd ..
conda activate llama
mv start_emo_api.py Emotion-LLaMA/
cd Emotion-LLaMA
python start_emo_api.py --port 5432
```
Run the test.
```sh
conda activate onuw
python test_emo_api.py
```

#### Setting OSUM API
```sh
conda create -n osum python=3.10
conda activate osum
pip install fastapi uvicorn python-multipart
git clone https://github.com/ASLP-lab/OSUM
cd OSUM
```
Follow the instructions in [ASLP-lab/OSUM](https://github.com/ASLP-lab/OSUM) to install dependencies, download the model, and set the path in the config.
```sh
cd ..
conda activate osum
mv start_osum_api.py OSUM/
cd OSUM
python start_osum_api.py --port 6543
```
Run the test.
```sh
conda activate onuw
python test_osum_api.py
```

Edit `configs/werewolf.json`.

Set `structure` in corresponding player's config to `human:mm` or `human:cli`.

Edit `onuw/mm_utils.py`, Line 11-12:
```python
# Emotion-LLaMA
EMOTION_API_URL = 'http://127.0.0.1:5432/analyze_emotion'
# OSUM
TRANSCRIPTION_API_URL = 'http://127.0.0.1:6543/transcribe'
```

Test API.
```sh
python onuw/mm_utils.py
```

Run the game.
```sh
python main.py --env Werewolf --num_runs <number of runs of different settings> --num_repeats <number of repeating runs in one setting> --random --cli --save_path <save path for game logs>
```