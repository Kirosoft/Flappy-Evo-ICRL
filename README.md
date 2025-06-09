# Flappy Evo ICRL (LLM-Guided Flappy Bird Evolution)

## Project Abstract

This project is an experiment in evolutionary in-context reinforcement learning. It demonstrates how a Large Language Model (LLM) can be used not only to generate and mutate policies, but also to guide the evolution of an agent's behavior in a simulated environment. The system evolves Flappy Bird-playing policies using a hybrid approach: traditional evolutionary algorithms are combined with LLM-guided mutations and policy generation. The LLM is prompted in-context with both successful and failed policies, enabling it to suggest creative, context-aware improvements that avoid known failure modes. This approach leverages the reasoning and generalization abilities of LLMs to accelerate and enrich the policy search process, resulting in more robust and effective agents than random evolution alone.

Note: This project was vibe coded!

## Features
- **Interactive Play:** Play Flappy Bird yourself using the keyboard.
- **AI Training:** Train an AI policy using evolutionary algorithms, with LLM-guided mutation and population seeding. Real-time training progress is visualized.
- **AI Playback:** Watch the best evolved policy play the game automatically.


## Project Structure
- `main.py` — Main application and UI loop
- `flappy_sim.py` — FlappyBirdGame class (game logic)
- `policy.py` — Policy normalization, validation, random generation, and training logic
- `evolution.py` — Evolutionary algorithm: fitness, crossover, mutation
- `llm.py` — LLM API calls and prompt construction
- `utils.py` — Utility functions (file I/O, logging)
- `game_instance.py` — Shared FlappyBirdGame instance for evaluation
- `prompts/` — Prompt templates and example policies for LLM
- `generation_best_policies/` — Per-generation best policies (saved during training)
- `docs/` — Additional documentation and screenshots

## Requirements
- Python 3.x
- [pygame](https://www.pygame.org/)
- [requests](https://docs.python-requests.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [Ollama](https://ollama.com/) (local LLM server, e.g., gemma3:14b)

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
1. **Configure Environment:**
   - Copy `.env.example` to `.env` and set the `OLLAMA_ENDPOINT` and `OLLAMA_MODEL` variables as needed.
   - Ensure Ollama is running and the required model is pulled.
2. **Run the Application:**
   ```
   python main.py
   ```
3. **Main Menu Options:**
   - Play Best Policy: Watch the best evolved policy play.
   - Train New Policy: Start a new LLM-guided evolutionary training run.
   - Play Yourself: Play Flappy Bird interactively.
   - Quit: Exit the application.

## Documentation
- See ![Documentation](docs/manual.md)

## LLM Integration
- The LLM is used to generate initial policy populations (Gemma3) and to perform intelligent mutation during evolution.
- If the LLM is unavailable, the system falls back to random policy generation/mutation.
- All LLM prompts and responses are logged for debugging.

## Troubleshooting
- If the LLM is not responding, check your `.env` settings and that Ollama is running.
- For UnboundLocalError in `get_coded_state`, ensure the `dist_val` assignment is correct (see `flappy_sim.py`).

## Screenshots
Screenshots and additional documentation can be found in the `docs/` folder:

- ![Main Menu](docs/Screenshot%202025-06-09%20000020.png)
- ![Sample Training Graph](docs/image.png)

See more images and explanations in the `docs/` folder.

## Credits
- Project inspired by LLM-guided reinforcement learning research.
- Uses Gemma for local LLM inference.

---
For more details, see the code comments, prompt templates in the `prompts/` directory, and the `docs/` folder for screenshots and manual.
