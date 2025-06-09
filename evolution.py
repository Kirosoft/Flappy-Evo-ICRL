import random
from llm import llm_intelligent_mutation
from policy import generate_random_state_key, get_action_from_policy

# --- Constants ---
DEFAULT_ACTIONS = ["flap", "do_nothing"]
FITNESS_SCORE_WEIGHT = 500
FITNESS_FRAMES_WEIGHT = 10
MUTATION_LLM_PROB = 0.33
MUTATION_REMOVE_PROB = 0.1
MUTATION_FLIP_PROB = 0.5

MAX_SIM_FRAMES = 4000


def evaluate_fitness(policy: dict, game_sim) -> int:
    """
    Evaluate the fitness of a policy by running it in the provided game simulator.
    Returns the fitness score as an integer.
    """
    game_sim.reset()
    action_map = {"flap": 1, "do_nothing": 0}
    if not (isinstance(policy, dict) and "default" in policy):
        pass
    game_over = False
    while not game_over:
        current_coded_state = game_sim.get_coded_state()
        numeric_action = get_action_from_policy(policy, current_coded_state, action_map)
        game_over = game_sim.step(numeric_action)
        if game_sim.frames_survived > MAX_SIM_FRAMES:
            break
    return (game_sim.frames_survived * FITNESS_FRAMES_WEIGHT) + (game_sim.score * FITNESS_SCORE_WEIGHT)


def crossover(parent1: dict, parent2: dict) -> dict:
    """
    Create a child policy by combining rules from two parent policies.
    """
    child_policy = {}
    keys = set(list(parent1.keys()) + list(parent2.keys()))
    for key in keys:
        if key == "default":
            continue
        chosen_val = None
        if key in parent1 and key in parent2:
            chosen_val = parent1[key] if random.random() < 0.5 else parent2[key]
        elif key in parent1:
            chosen_val = parent1[key]
        elif key in parent2:
            chosen_val = parent2[key]
        if chosen_val is not None:
            child_policy[key] = chosen_val
    child_policy["default"] = (
        parent1.get("default", "flap") if random.random() < 0.5 else parent2.get("default", "flap")
    )
    return {k: v for k, v in child_policy.items() if v is not None}


def mutate(policy: dict, use_llm: bool = False) -> dict:
    """
    Mutate a policy, optionally using the LLM for intelligent mutation.
    Returns a new mutated policy dict.
    """
    if use_llm and random.random() < MUTATION_LLM_PROB:
        llm_mutated = llm_intelligent_mutation(policy.copy())
        if llm_mutated:
            return llm_mutated
    new_policy = policy.copy()  # Fallback to random
    mutation_type = random.random()
    if mutation_type < MUTATION_REMOVE_PROB and len(new_policy) > 1:
        keys = [k for k in new_policy.keys() if k != "default"]
        if keys:
            del new_policy[random.choice(keys)]
    elif mutation_type < MUTATION_FLIP_PROB:
        key_to_mutate = random.choice(list(new_policy.keys()))
        new_policy[key_to_mutate] = (
            "flap" if new_policy[key_to_mutate] == "do_nothing" else "do_nothing"
        )
    else:
        new_state_key = generate_random_state_key(random.randint(1, 3))
        if new_state_key and new_state_key not in new_policy:
            new_policy[new_state_key] = random.choice(DEFAULT_ACTIONS)
        elif "default" in new_policy:
            new_policy["default"] = (
                "flap" if new_policy["default"] == "do_nothing" else "do_nothing"
            )
    if "default" not in new_policy:
        new_policy["default"] = random.choice(DEFAULT_ACTIONS)
    return new_policy
