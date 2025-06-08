## Flappy Evo Code Specification

### 1. Overview

`Flappy Evo` is a Python-based application that evolves control policies for a simplified Flappy Bird game using an evolutionary algorithm, optionally guided by an LLM (via Ollama). It provides:

* A simulated Flappy Bird environment (`FlappyBirdGame`).
* Policy representation, normalization, validation, and action-selection logic.
* Integration with an LLM for generating an initial policy population and intelligent mutations.
* An evolutionary training loop with fitness evaluation, selection, crossover, mutation, and logging.
* Real-time visualization of training progress via Pygame.
* A menu-driven interface to train, play with the evolved policy, or play interactively.

### 2. Configuration & Environment

* **Environment Variables** (loaded via `dotenv`):

  * `POLICY_FORMAT_GUIDANCE_PATH`: Path to policy format guidance template.
  * `EXAMPLE_POLICY_PATH`: Path to example policy JSON.
  * `INITIAL_POP_TEMPLATE_PATH`: Path to initial population prompt template.
  * `MUTATION_TEMPLATE_PATH`: Path to mutation prompt template.
  * `OLLAMA_ENDPOINT`: URL of the Ollama API endpoint (default: `http://localhost:11434/api/generate`).
  * `OLLAMA_MODEL`: LLM model identifier (default: `qwen3:!4b`).
* **Constants & Settings**:

  * Game parameters: pipe speed, gap size ranges, bird physics (`GRAVITY`, `FLAP_STRENGTH`), screen dimensions.
  * LLM parameters: `LLM_TIMEOUT`, `LLM_TEMPERATURE_INIT_POP`, `LLM_TEMPERATURE_MUTATION`.
  * Paths: `FAILED_LLM_LOG_FILE`, `PER_GENERATION_BEST_DIR`.

### 3. Dependencies

* `pygame`: Game simulation, rendering, and GUI.
* `requests`: HTTP requests to the Ollama API.
* `json`, `csv`, `time`, `random`, `os`, `threading`, `shutil`: Standard library utilities.
* `dotenv`: Loading environment variables.

### 4. Core Game Mechanics

#### 4.1 Class: `FlappyBirdGame`

* **Attributes**: Bird state (`bird_y`, `bird_velocity`), pipe position and gap, score, frame count, physical constants.
* **Methods**:

  * `__init__()`: Initializes a new game instance with random pipe gap.
  * `_random_gap_y(gap_height)`: Calculates a valid vertical position for the pipe gap.
  * `reset()`: Re-initializes the game to starting state.
  * `step(action) -> bool`: Advances the game by one frame with `action` (0 or 1). Returns `True` if collision occurs.
  * `get_coded_state() -> str`: Encodes the current game state into a discrete key of the form `dist:..._pos:..._velo:...`.

### 5. LLM Integration

#### 5.1 LLM API Calls

* **`call_ollama_blocking(prompt, temperature) -> (parsed_json, raw_text)`**

  * Sends a blocking request to the Ollama API.
  * Parses JSON from a `json` code fence in the response.
  * Handles errors (timeout, connection, JSON decode) by logging failures and returning `None`.
* **`call_ollama_blocking_for_thread(prompt, temperature, result_container, error_container, raw_response_container)`**

  * Wrapper for threaded LLM calls, storing results in shared containers.

#### 5.2 Prompt Generators

* **`llm_generate_initial_population_prompt_str(n, failed_policies)`**: Formats the initial population prompt using loaded templates and examples.
* **`llm_intelligent_mutation(policy) -> dict or None`**: Generates a mutated policy via the LLM, normalizes and validates it.

### 6. Policy Normalization & Validation

#### 6.1 `normalize_policy(policy_data_input, expect_list=False)`

* **Purpose**: Convert raw LLM output into a well-structured policy dict or list of dicts.
* **Process**:

  * Unwrap nested objects (e.g., `{"policy": {...}}`).
  * Ensure a valid `default` action (`"flap"` or `"do_nothing"`).
  * Correct common typos (`"do_no'thing"`).
  * Filter out invalid rules and keys, sort multi-part keys alphabetically.
* **Return**: A single dict (if `expect_list=False`) or a list of dicts (if `expect_list=True`).

#### 6.2 `validate_policy(policy) -> bool`

* **Checks**:

  * `policy` is a dict with a valid `default` action.
  * All rule values are valid actions.
  * Keys follow the `prefix:value` format for `pos:`, `dist:`, `velo:`, without duplicates.
  * Multi-part keys are alphabetically sorted.

### 6.3 Policy Execution & Rule Matching

When the game needs an action each frame, the policy is applied as follows:

1. **Full-State Lookup (3-part)**

   * Compute the current coded state: three parts (`dist:…`, `pos:…`, `velo:…`), sorted alphabetically and joined with underscores.
   * If a matching 3-part key exists in the policy, use its action.
2. **Partial-State Fallback (2-part)**

   * Otherwise, try every combination of two of the three descriptors (also sorted). First found wins.
3. **Single-Part Fallback (1-part)**

   * If no 2-part rule matches, check each single descriptor in turn.
4. **Default Action**

   * If none of the above match, use the mandatory `"default"` action.

#### Example Walkthrough

Given the bird is “medium” away, “aligned” with the gap, and “rising”:

* Sorted key: `dist:medium_pos:aligned_velo:rising`
* No exact 3-part entry → try 2-part keys:

  * `dist:medium_pos:aligned` (not present)
  * `dist:medium_velo:rising` → **found** → action **"flap"**.
* Even though `"dist:medium": "do_nothing"` exists, the 2-part rule takes precedence.

This layered approach—3-part first, then 2-part, then 1-part, then default—lets a compact policy cover both specific and general situations.

* **Checks**:

  * `policy` is a dict with a valid `default` action.
  * All rule values are valid actions.
  * Keys follow the `prefix:value` format for `pos:`, `dist:`, `velo:`, without duplicates.
  * Multi-part keys are alphabetically sorted.

### 7. Evolutionary Algorithm

#### 7.1 Policy Representation & Random Generation

* **`generate_random_policy() -> dict`**: Creates a random policy with `default` and up to two random state rules.
* **`generate_random_state_key(num_parts) -> str`**: Builds a state key of length `num_parts` from `pos`, `dist`, `velo` values.

#### 7.2 Genetic Operators

* **`get_action_from_policy(policy, full_coded_state, action_map) -> int`**: Retrieves an action (0/1) from a policy given the current state, with fallback to partial matches and default.
* **`evaluate_fitness(policy, game_sim) -> float`**: Plays a full game simulation and returns a fitness score based on frames survived and pipes passed.
* **`crossover(parent1, parent2) -> dict`**: Combines two parent policies by randomly selecting rules from each.
* **`mutate(policy, use_llm=False) -> dict`**: Applies random mutation or LLM-guided mutation to a policy, ensuring validity.

##### Mutation Details & Adaptation

The `mutate` function introduces diversity while leveraging LLM guidance to adapt mutations over time:

1. **LLM-Guided Mutation** (≈33% chance if `use_llm` is `True`):

   * The current policy JSON is sent to the LLM along with the mutation prompt template and recent failure examples.
   * The LLM proposes a single, context-aware change: adding, modifying, or removing one rule (or changing the default action).
   * The returned policy is normalized and validated; successful mutations replace the original, steering search toward promising modifications.
   * Failed LLM outputs (invalid JSON, formatting errors, or invalid policy rules) are logged and fed back into future prompts, improving LLM adaptation.

2. **Random Mutation** (fallback or when LLM is not used):

   * **Rule Deletion** (10% chance): Remove a randomly chosen non-default rule, simplifying the policy.
   * **Rule Flip** (40% chance): Toggle the action (`"flap"`⇄`"do_nothing"`) of a randomly selected rule or the default action if no other rules.
   * **Rule Addition** (remaining chance): Generate a new random state key (`generate_random_state_key`) and assign it a random action, increasing policy specificity.
   * Ensures a `"default"` key always exists by reassigning if deleted unintentionally.

3. **Adaptive Feedback Loop**:

   * Policies that fail validation post-mutation are logged in `failed_policies_for_llm`.
   * These failures are included in subsequent LLM prompts, allowing the LLM to learn from past mistakes and gradually improve its mutation suggestions.
   * Over generations, the proportion of effective, meaningful LLM-driven mutations increases, enhancing convergence speed and policy quality.

#### 7.3 Training Loop: `train_policy()` Training Loop: `train_policy()`

* **Stages**:

  1. **Initialization**:

     * Create CSV log and per-generation best policy directory.
     * Generate initial population: half from LLM, half random fill.
  2. **Evolution (for ****`NUM_GENERATIONS`****)**:

     * Evaluate fitness of each policy.
     * Log fitness and policies to CSV.
     * Select top `NUM_PARENTS`, apply crossover and mutation to form next generation.
     * Track overall best policy and save per-generation best to disk.
     * Visualize progress (fitness curve, best policy) via Pygame.
  3. **Completion**:

     * Save `best_policy.json` to disk.
     * Return the best policy.

### 8. GUI & Menu

#### 8.1 Training Visualization

* Uses Pygame to draw axes, fitness curve, generation counters, and display the current overall best policy.

#### 8.2 Main Menu: `main_menu()`

* **Options**:

  * Play Best Policy (if `best_policy.json` exists and is valid).
  * Train New Policy.
  * Play Yourself (interactive human play).
  * Quit.
* **Controls**:

  * Up/Down arrows to navigate.
  * Enter to select.
* Dynamically rebuilds menu based on the presence and validity of a saved policy.

### 9. Entry Point

* Running the script (`if __name__ == "__main__"`) launches `main_menu()`, handling initialization, command dispatch, and cleanup (`pygame.quit()`).

### 10. In-Context Reinforcement Learning (ICRL)

ICRL in Flappy Evo refers to the seamless integration of in-context guidance from a large language model into the evolutionary training loop, effectively blending evolutionary search with LLM-driven policy refinement:

* **Context Provision**: The LLM receives structured prompt templates (initial population and mutation templates) containing:

  * **Policy Format Guidance**: Clear formatting rules to ensure JSON validity and consistent state-action mapping.
  * **Example Policies**: Small, correctly formatted examples to demonstrate expected output structure.
  * **Failure History**: Recent policies that failed validation, allowing the LLM to *learn* from past mistakes.

* **LLM-Driven Seeding**: For the initial population, the LLM produces multiple candidate policies in one shot, informed by the example and format guidance, reducing reliance on purely random initialization.

* **Intelligent Mutations**: During mutation, a subset of offspring are sent to the LLM along with their current policy JSON. The LLM applies focused, one-off edits (add/change/remove a single rule or alter the default) that adhere to the policy rules, guiding the search toward promising regions of the policy space.

* **Feedback Loop**: Policies that fail normalization or validation are fed back into subsequent prompts, enabling the LLM to iteratively improve its outputs. This forms a closed loop where the LLM refines its in-context understanding of the task over generations.

* **Advantages of ICRL**:

  1. **Sample Efficiency**: By leveraging the LLM’s learned priors and context, the algorithm finds higher-quality policies faster than pure random search.
  2. **Adaptivity**: The LLM adapts its mutations based on explicit format rules and recent failures, reducing repeated mistakes.
  3. **Simplicity**: No additional ML training is required; the LLM operates purely via well-engineered prompts.

In Flappy Evo, ICRL augments the classic genetic algorithm with language-based in-context learning, yielding a hybrid approach that accelerates discovery of robust control policies.

### 11. Experimental Results

We performed a detailed analysis on generation **66** of one training run, examining the fitness of policies with indices **6** through **19**. The following figure shows how fitness varies across those policies.

*Figure 1: Fitness vs. Policy Index for Generation 66.*

Below is the plotted distribution of fitness values.

| Policy Index | Fitness |
| ------------ | ------- |
| 6            | 1 410   |
| 7            | 1 410   |
| 8            | 42 560  |
| 9            | 1 420   |
| 10           | 56 010  |
| 11           | 37 310  |
| 12           | 360     |
| 13           | 3 120   |
| 14           | 320     |
| 15           | 980     |
| 16           | 54 550  |
| 17           | 3 120   |
| 18           | 40 860  |
| 19           | 56 010  |

**Detailed Summary:**

* **Total policies evaluated:** 14
* **Fitness range:** 320 – 56 010
* **Mean fitness:** 21 388.6
* **Median fitness:** 3 120
* **High-performers (≥ 40 000):** 5/14 (36%) → indices 8, 10, 16, 18, 19
* **Low-performers (≤ 1 000):** 3/14 (21%) → indices 12, 14, 15

These results reveal a strongly right-skewed distribution: the majority of policies cluster at lower fitness, while a small subset achieves exceptionally high scores, indicating that the evolutionary process is efficiently discovering standout policies that drive overall progress.

### 12. Training Curve

Next, we examine how the **best fitness** evolves across **all 66 generations** of this run. The plot below shows the maximum fitness reached at each generation.

*Figure 2: Best Fitness vs. Generation (1–66).*
![Training Curve Example](./image.png)

From the training curve, we observe:

* **Rapid early gains**: Fitness jumps from \~10 000 in generation 1 to \~55 000 by generation 5, driven by LLM seeding and strong early mutations.
* **Plateau phases**: Once a high-fitness policy is found (\~55 000), the algorithm spends many generations refining but rarely improving beyond that ceiling, indicating a local optimum.
* **Transient dips**: Occasionally (e.g., generations 8, 22, 34, 39, 51, 65), the best policy drops by 10–20 000 fitness, suggesting exploration re-introducing diversity.
* **Final stability**: By generation 50 onward, best fitness hovers tightly around \~55 000, showing convergence to a robust policy.

Overall, the hybrid ICRL + evolutionary approach yields fast convergence to high-quality solutions, with controlled exploration preventing premature stagnation.

### 13. Conclusion

Flappy Evo successfully demonstrates how combining genetic algorithms with LLM-driven in-context reinforcement learning (ICRL) can accelerate and stabilize the search for robust Flappy Bird control policies. Key takeaways:

* **Efficiency of LLM Seeding**: Incorporating LLM-generated initial policies rapidly jumps the population to high-fitness regions, reducing random search overhead.
* **Targeted Mutations**: Intelligent, single-rule edits from the LLM adapt over time by learning from past failures, boosting sample efficiency compared to purely random mutations.
* **Evolutionary Robustness**: The GA framework—selection, crossover, mutation—provides exploration and exploitation balance, ensuring convergence while escaping local optima via controlled dips.
* **Empirical Results**: Across 66 generations, Flappy Evo reached fitness levels > 55 000 within the first few generations and maintained performance with minimal drift, highlighting both speed and stability.

This specification and accompanying experiments illustrate Flappy Evo’s potential as a template for hybrid neuro-symbolic optimization: merging algorithmic search with powerful language-model priors to solve discrete control tasks effectively.

### 14. Future Improvements

To further enhance learning efficiency and policy quality, consider the following directions:

1. **Curriculum Learning**: Gradually increase difficulty by starting with wider pipe gaps or slower pipe speeds, then narrow the gap or speed up pipes as policies improve. This staged curriculum can guide the population from easy to hard scenarios, boosting stability and learning speed.

2. **Diverse Objective Functions**: Instead of a single fitness metric, introduce multi-objective optimization (e.g., minimize sudden altitude changes, maximize smoothness) to evolve more human‐like and robust behaviors. Use Pareto fronts to select policies balancing multiple criteria.

3. **Adaptive Mutation Rates**: Dynamically adjust mutation probabilities based on population diversity or convergence rate. For example, increase mutation when the fitness plateau stagnates, and reduce when rapid gains are observed.

4. **Ensemble Policies**: Maintain a small ensemble of top-performing policies and combine their actions via voting or confidence-weighted averaging. This can mitigate overfitting to specific pipe sequences and improve generalization.

5. **State Abstraction Enhancement**: Incorporate additional state features (e.g., bird’s distance from top/bottom bounds, velocity magnitude quantiles) or discretize continuous values more finely. Richer state encoding can allow nuanced rule development.

6. **Replay Buffer & Offline Refinement**: Collect game trajectories from high-performing policies into a replay buffer. Use offline policy evaluation or fine-tuning (e.g., Q‐learning, policy gradients) on these trajectories to locally optimize rule weights.

7. **Hybrid Neural‐Symbolic Policies**: Instead of purely symbolic rules, augment policies with small neural networks for action weighting, where symbolic rules propose candidates and the network refines decisions. This can capture complex interactions beyond simple rule conjunctions.

8. **Cross‐Population Exchange**: Periodically merge individuals from independently seeded populations (with different random seeds) to inject novel genetic material and prevent premature convergence.

9. **Meta‐Learning for Prompt Engineering**: Automate prompt template refinement via outer‐loop optimization—treat prompt variations (template wording, guidance emphasis) as hyperparameters and evolve them to maximize policy quality.

10. **Real‐Time Human-in-the‐Loop Feedback**: Allow occasional human evaluation of policy snippets to guide the LLM’s in-context examples, combining human intuition with automated evolution.

Implementing one or more of these strategies can lead to faster convergence, richer policy behaviors, and greater robustness across diverse Flappy Bird scenarios.

### 15. Literature Review

Recent work has explored the use of Large Language Models (LLMs) to **guide the search process** in reinforcement learning and evolutionary algorithms, often through in-context learning (ICL) rather than weight updates:

* **EvoLLM (Lange et al., 2024)** demonstrates zero-shot black-box optimization by giving an LLM a sorted list of candidates and prompting it to propose improved solutions, effectively replacing crossover/mutation operators and outperforming random search on benchmark functions.

* **EvoPrompt (Guo et al., 2024)** evolves text prompts via LLM-generated mutations, achieving up to 25% gains on difficult language tasks compared to human-engineered prompts, highlighting synergy between LLM priors and evolutionary exploration.

* **Code-as-Policies (Google PaLM, 2023)** showcases neuro-symbolic policy generation, where an LLM outputs Python control code for robotic tasks with superior generalization and interpretability over end-to-end neural policies.

* **Voyager (Wang et al., 2023)** iteratively uses GPT-4 to write, execute, and refine game-playing code in Minecraft, collecting feedback in each loop to build a library of reusable, interpretable skills that outperform traditional RL agents by 3×–8× on exploration and milestone achievement.

* **μMOEA (Tian et al., 2025)** integrates LLMs into a multi-objective evolutionary algorithm, using LLM-generated seed populations and LLM-informed mutations to improve search efficiency and solution diversity over state-of-the-art EAs.

* **LLM-POET (Aki et al., 2024)** replaces parametric environment mutation in the POET framework with an LLM that describes new procedural environments, yielding 34% more diverse and challenging scenarios that accelerate agent learning.

**Key Comparisons to Flappy Evo:**

1. **LLM-Driven Seeding & Mutation:** Like EvoLLM and μMOEA, Flappy Evo uses LLMs to generate initial policies and propose focused single-rule edits, leveraging pre-trained knowledge to accelerate convergence.

2. **Purely Symbolic Policies:** In line with Code-as-Policies and Voyager’s code-based skills, Flappy Evo’s JSON rule sets are interpretable, human-auditable, and robust, avoiding black-box neural policies.

3. **In-Context Feedback:** Similar to Voyager’s execution feedback loop, Flappy Evo feeds failed or suboptimal policies back into prompts, enabling the LLM to adapt mutation quality over generations.

4. **Performance & Sample Efficiency:** Across various domains, LLM-guided EA methods consistently achieve rapid early gains and maintain robustness. Flappy Evo’s ability to reach >55 000 fitness within \~5 generations mirrors speedups seen in other LLM-guided frameworks.

**Open Questions & Research Gaps:**

* Scaling symbolic ICL-EA methods to high-dimensional or continuous domains.
* Balancing LLM guidance with random exploration to avoid prior-induced blind spots.
* Developing dynamic prompting and memory strategies to sustain long-horizon learning.
* Ensuring safety and validity of LLM-proposed policies via formal checks or self-audit.
* Theoretical understanding of how transformer-based inference approximates optimization steps.

Flappy Evo stands alongside these recent advances, illustrating the promise and challenges of hybrid LLM + evolutionary strategies for discrete control tasks.
