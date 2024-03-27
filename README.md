# DOOM Mistral
<p align="center">
<img src = https://github.com/umuthopeyildirim/DOOM-Mistral/assets/90144938/0259a48f-c51d-4223-9ffe-f626dfcb2c73) width="18%" height="9%">
</p>
<p align="center">
<img src = https://github.com/umuthopeyildirim/DOOM-Mistral/assets/90144938/e82098e9-2401-48d9-b99d-1e78f2cc8ba7)>
</p>


This repository is home to the DOOM Mistral project, crafted during the CV [Mistral AI Hackathon](https://cerebralvalley.notion.site/Mistral-AI-Hackathon-Event-Details-Hackers-ee95c2545eda4ce1ae82bd5910a4a3ae) in San Francisco, on March 23-24 2024. Our team (Bhav, Umut, and Paul) developed a deep learning model capable of playing DOOM through visual input alone, utilizing the [ViZDoom](https://vizdoom.farama.org/#) engine, a prominent tool in visual reinforcement learning. We transformed each game frame into a 32x64 grid, representing game elements in each cell, enabling our model to interpret the game visually. For a glimpse into how the model views the game, refer to this video (note: contains strobing effects):

https://twitter.com/i/status/1772075251297550457

We generated training data by playing DOOM, then fine-tuned a LoRA model on `Mistral-7B`, achieving reasonable results. While not an expert, the model can navigate the map and engage enemies when they appear.

Post-training, the model's actions were integrated back into the game, adopting a simplified RL approach due to the hackathon's time constraints.

![DOOM Mistral Visualization](./docs/arch.jpeg)

Watch the model in action here:

https://twitter.com/i/status/1772166935410532709

You can find our Weights & Biases dashboard [here](https://wandb.ai/hopesweaty/doom-mistral).

## Setting Up DOOM Mistral

To initialize, create a virtual environment and install the ViZDoom dependencies:

```bash
pip install -r requirements.txt
```

Rename `.env.example` to `.env` and input your `FIREWORKS_API_KEY`.

To start the game, execute:

```bash
python llm_game.py
```

## Generating Training Data

Capture gameplay for training with:
`python user_game.py`

Access the training dataset at [HuggingFace 🤗](https://huggingface.co/datasets/CV-Mistral-Hackathon/doom-mistral-final), download it, and place it in the root directory.

Upload your dataset to Fireworks.ai for model training:

```bash
firectl create dataset doom-mistral doom_mistral.jsonl
```

## Model Training and Usage

Training settings are listed in `train_settings.yaml`. To train, use:

```bash
firectl create fine-tuning-job --settings-file train_settings.yaml --display-name "DOOM-Mistral"
```

Access the trained model on fireworks.ai, named `doom-mistral`.

To play the game with any LLM, modify `model_id` in line 16, then run:
`python llm_game.py`

### Understanding the Grid

- E: Enemy
- P: Player
- B: Bullet
- W: Wall
- F: Floor
- A: ArmorBonus
- Z: Zombieman
- H: HealthBonus
- S: Stimpack

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=umuthopeyildirim/DOOM-Mistral&type=Date)](https://star-history.com/#umuthopeyildirim/DOOM-Mistral&Date)

## Licensing

ViZDoom's original code is under the MIT license. ZDoom incorporates various sources, each with [distinct licensing](http://zdoom.org/wiki/license).
