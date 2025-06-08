<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->


Specification: LLM-Guided AI Trainer for Flappy Bird
Version: 1.1
Date: June 8, 2025

1. Overview
This document specifies the design and architecture for an interactive Python application that serves as both a playable Flappy Bird game and a development environment for an advanced AI agent.

The primary purpose of the application is to demonstrate and test a hybrid AI training methodology called Language-Guided Evolutionary Reinforcement Learning. This approach combines a traditional evolutionary algorithm with the high-level reasoning capabilities of a Large Language Model (LLM) to "evolve" an effective policy for playing the game.

The application will feature three primary modes of operation, accessible via an interactive Pygame-based main menu:

Interactive Play: Allows a human user to play the game via keyboard input.

AI Training: Initiates an evolutionary training process, using a local LLM to guide learning, and provides real-time visual feedback on the training progress.

AI Playback: Loads the best policy generated during a training run and allows the user to watch the trained AI agent play the game autonomously.

2. Core Components
The application is designed with a clean separation between the game logic and the controlling agent/application shell.

2.1. Game Engine (FlappyGame Class)
This is a self-contained Python class that manages the complete state and rules of the Flappy Bird simulation. It has no knowledge of Pygame, the LLM, or the training algorithm.

Attributes:

bird_y, bird_velocity: Manages the bird's vertical physics.

pipes: A list of dictionaries, each representing a pipe with its position and state.

score, frames_survived: Tracks performance metrics.

is_game_over: A boolean flag indicating the end of a run.

Methods:

reset(): Restores all game variables to their initial state for a new game.

step(action): The core update function. It accepts a single integer (1 for flap, 0 for no action) and advances the game simulation by one frame. It handles all physics, collision detection, and scoring logic.

get_state(): Returns a dictionary of the complete, raw numerical state of the game.

get_coded_state(): The "state tokenization" layer. This crucial method translates the raw numerical state into a discrete, human-readable string (e.g., "pos:below_dist:close_velo:falling"). This coded state is used as the key for the AI policies.

2.2. Main Application (App Class)
This class manages the application window, user interface, main loop, and mode switching. It acts as the orchestrator for all other components.

State Management: Manages the current application state (MAIN_MENU, PLAYING_HUMAN, TRAINING, PLAYING_AI).

Rendering: Handles all drawing to the screen using Pygame, including the main menu buttons, the game elements, and the dynamic training dashboard.

Input Handling: Processes all user input (mouse clicks and key presses) and directs actions based on the current application state.

Mode Logic: Contains the specific update loops for each mode of operation.

3. Functional Modes
3.1. Mode 1: Interactive Play
Trigger: User selects "1. Play Game" from the main menu.

Control: The game loop listens for pygame.K_SPACE keydown events. A key press sends action=1 to the game.step() method. In all other frames, action=0 is sent.

Goal: To provide a standard, playable version of the game for humans.

3.2. Mode 2: AI Training
Trigger: User selects "2. Train AI Policy" from the main menu.

Workflow:

Initialization (start_training):

The system makes an initial API call to the local LLM (llm_generate_initial_population) to generate a diverse population of starting policies. This seeds the evolution with sensible, varied strategies.

If the LLM call fails, the system falls back to a randomly generated population to ensure operation continues.

Generational Loop (run_one_training_generation): The system iterates through a set number of generations. In each generation:
a. Evaluation: Every policy in the current population is evaluated by playing a full game (headless, no visualization) to get a fitness score. The fitness function is (score * 500) + (frames_survived * 10).
b. Selection: The top-performing policies (parents) are selected based on their fitness scores. The worst-performing policies are cached to be used as negative examples.
c. Reproduction: A new generation is created. The best parents are carried over directly (elitism). The remaining slots are filled by creating "children" through a crossover of two parents.
d. Mutation: Each child policy undergoes mutation. There is a configured chance (LLM_MUTATION_CHANCE) that the mutation will be performed by the llm_intelligent_mutation function. This function's prompt includes the successful parent policy as well as the cached list of failed policies, instructing the LLM to suggest a creative improvement while avoiding known failure modes. Otherwise, a simple random mutation is applied as a fallback.

Completion: After the final generation, the best policy ever found is saved to a file named best_policy.json.

Visual Feedback: During training, the screen displays a real-time matplotlib graph of the best fitness score per generation, allowing the user to see the AI's learning progress. It also displays the JSON representation of the best policy found so far.

3.3. Mode 3: AI Playback
Trigger: User selects "3. Watch AI Play" from the main menu.

Workflow:

The application first attempts to load the policy from best_policy.json.

If the file does not exist, an error message is displayed on the menu.

If successful, the game starts. In each frame, the application calls game.get_coded_state() to get the current situation, looks up the corresponding action in the loaded policy dictionary, and sends that action to the game.step() method.

Goal: To allow the user to observe the performance and behavior of the final, fully trained agent.

4. Technical Requirements
Python 3.x

Libraries:

pygame: For game graphics and user interaction.

requests: For making HTTP API calls to the local LLM.

matplotlib: For rendering the training progress graph.

numpy: A dependency of matplotlib.

python-dotenv: For managing environment variables from a .env file.

Local LLM Server:

Requires a running instance of Ollama.

Requires a powerful, instruction-following model to be pulled (e.g., llama3:8b-instruct is recommended for its JSON mode adherence). The server must be accessible via the endpoint defined in the .env file.
