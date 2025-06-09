import random
from llm import llm_intelligent_mutation
from policy import generate_random_state_key, get_action_from_policy

def evaluate_fitness(policy, game_sim):
    game_sim.reset(); action_map = {"flap": 1, "do_nothing": 0}
    if not (isinstance(policy, dict) and "default" in policy): pass
    game_over = False
    while not game_over:
        current_coded_state = game_sim.get_coded_state()
        numeric_action = get_action_from_policy(policy, current_coded_state, action_map)
        game_over = game_sim.step(numeric_action)
        if game_sim.frames_survived > 4000: break
    return (game_sim.frames_survived * 10) + (game_sim.score * 500)

def crossover(parent1, parent2):
    child_policy = {}; keys = set(list(parent1.keys()) + list(parent2.keys()))
    for key in keys:
        if key == "default": continue
        chosen_val = None
        if key in parent1 and key in parent2: chosen_val = parent1[key] if random.random() < 0.5 else parent2[key]
        elif key in parent1: chosen_val = parent1[key]
        elif key in parent2: chosen_val = parent2[key]
        if chosen_val is not None: child_policy[key] = chosen_val
    child_policy["default"] = parent1.get("default", "flap") if random.random() < 0.5 else parent2.get("default", "flap")
    return {k:v for k,v in child_policy.items() if v is not None}

def mutate(policy, use_llm=False):
    if use_llm and random.random() < 0.33:
        llm_mutated = llm_intelligent_mutation(policy.copy()) 
        if llm_mutated: return llm_mutated
    new_policy = policy.copy() # Fallback to random
    mutation_type = random.random()
    if mutation_type < 0.1 and len(new_policy) > 1:
        keys = [k for k in new_policy.keys() if k != "default"];
        if keys: del new_policy[random.choice(keys)]
    elif mutation_type < 0.5:
        key_to_mutate = random.choice(list(new_policy.keys()))
        new_policy[key_to_mutate] = "flap" if new_policy[key_to_mutate] == "do_nothing" else "do_nothing"
    else:
        new_state_key = generate_random_state_key(random.randint(1, 3))
        if new_state_key and new_state_key not in new_policy:
            new_policy[new_state_key] = random.choice(["flap", "do_nothing"])
        elif "default" in new_policy:
             new_policy["default"] = "flap" if new_policy["default"] == "do_nothing" else "do_nothing"
    if "default" not in new_policy: new_policy["default"] = random.choice(["flap", "do_nothing"])
    return new_policy