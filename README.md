### Installation

```sh
conda create -n onuw python=3.10
conda activate onuw
pip install -r requirements.txt
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
```bash
python main.py --env Werewolf --num_runs <number of runs of different settings> --num_repeats <number of repeating runs in one setting> --random --cli --save_path <save path for game logs>
```
The following options are only enabled when added:

- `random`: Randomly assign roles at the beginning
- `cli`: Launch cli (the interactive interface in the command line)

### Human Participation
Edit `configs/werewolf.json`.

Set `structure` in corresponding player's config to `human`.
