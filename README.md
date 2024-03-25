# GameCopilot

## Training data
To record games as training data, use:
`python examples/python/user_input.py`

The dataset used to train the final model is available at: 
https://huggingface.co/datasets/CV-Mistral-Hackathon/doom-mistral-final

The hyperparameters used are specified in train_settings.yaml

## Model
The final model is available on fireworks.ai under the name `doom-mistral-fixed-prompt-lr-assistant-fast-3`

To use an LLM to play the game, use:
`python examples/python/llm_game.py`


### Grid legend
- E - Enemy
- P - Player
- B - Bullet
- W - Wall
- F - Floor
- A - ArmorBonus
- Z - Zombieman
- H - HealthBonus
- S - Stimpack
