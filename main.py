import os
from dotenv import load_dotenv
import random
import json
import requests
import pygame
import csv
import time

load_dotenv()

# --- Gameplay Settings ---
PIPE_START_X_SETTING = 500
PIPE_SPEED_SETTING = 3
PIPE_RESPAWN_X_SETTING = 300
PIPE_GAP_MIN_SETTING = 130
PIPE_GAP_MAX_SETTING = 170
BIRD_Y_MIN_SETTING = 0
BIRD_Y_MAX_SETTING = 100
GRAVITY_SETTING = 0.10
FLAP_STRENGTH_SETTING = -1.50
PIPE_MARGIN_Y_SETTING = 50
# --- End Gameplay Settings ---

class FlappyBirdGame:
    PIPE_START_X = PIPE_START_X_SETTING
    PIPE_SPEED = PIPE_SPEED_SETTING
    PIPE_RESPAWN_X = PIPE_RESPAWN_X_SETTING
    PIPE_GAP_MIN = PIPE_GAP_MIN_SETTING
    PIPE_GAP_MAX = PIPE_GAP_MAX_SETTING
    BIRD_Y_MIN = BIRD_Y_MIN_SETTING
    BIRD_Y_MAX = BIRD_Y_MAX_SETTING
    GRAVITY = GRAVITY_SETTING
    FLAP_STRENGTH = FLAP_STRENGTH_SETTING
    PIPE_MARGIN_Y = PIPE_MARGIN_Y_SETTING

    GAME_SCREEN_HEIGHT = 600
    GAME_BIRD_PIXEL_RADIUS = 20
    GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE = GAME_SCREEN_HEIGHT - (2 * GAME_BIRD_PIXEL_RADIUS)

    def __init__(self):
        self.bird_y = 50
        self.bird_velocity = 0
        self.gravity = self.GRAVITY
        self.flap_strength = self.FLAP_STRENGTH
        self.pipe_x = self.PIPE_START_X
        self.score = 0
        self.frames_survived = 0
        self.pipe_width = 60
        self.gap_height = random.randint(self.PIPE_GAP_MIN, self.PIPE_GAP_MAX)
        self.pipe_gap_y = self._random_gap_y(self.gap_height)

    def _random_gap_y(self, gap_height_pixels):
        min_center_y = self.PIPE_MARGIN_Y + gap_height_pixels // 2
        max_center_y = self.GAME_SCREEN_HEIGHT - self.PIPE_MARGIN_Y - gap_height_pixels // 2
        if min_center_y >= max_center_y:
            return self.GAME_SCREEN_HEIGHT // 2
        return random.randint(min_center_y, max_center_y)

    def reset(self):
        self.__init__()

    def step(self, action):
        self.frames_survived += 1
        if action == 1:
            self.bird_velocity = self.flap_strength
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.bird_y = max(self.BIRD_Y_MIN, min(self.BIRD_Y_MAX, self.bird_y))
        self.pipe_x -= self.PIPE_SPEED
        if self.pipe_x + self.pipe_width < 0:
            self.pipe_x = self.PIPE_RESPAWN_X
            self.gap_height = random.randint(self.PIPE_GAP_MIN, self.PIPE_GAP_MAX)
            self.pipe_gap_y = self._random_gap_y(self.gap_height)
            self.score += 1
        is_collision = False
        bird_px_x_center = 60
        bird_px_radius = self.GAME_BIRD_PIXEL_RADIUS
        bird_px_left = bird_px_x_center - bird_px_radius
        bird_px_right = bird_px_x_center + bird_px_radius
        bird_px_y_center = (self.bird_y / self.BIRD_Y_MAX) * self.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE
        bird_px_top = bird_px_y_center - bird_px_radius
        bird_px_bottom = bird_px_y_center + bird_px_radius
        pipe_px_left = self.pipe_x
        pipe_px_right = self.pipe_x + self.pipe_width
        gap_px_top = self.pipe_gap_y - self.gap_height // 2
        gap_px_bottom = self.pipe_gap_y + self.gap_height // 2
        if bird_px_right > pipe_px_left and bird_px_left < pipe_px_right:
            if bird_px_top < gap_px_top or bird_px_bottom > gap_px_bottom:
                is_collision = True
        if self.bird_y >= self.BIRD_Y_MAX or self.bird_y <= self.BIRD_Y_MIN:
            is_collision = True
        return is_collision

    def get_coded_state(self): # Returns the canonical, sorted 3-part state string
        bird_px_y_center = (self.bird_y / self.BIRD_Y_MAX) * self.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE
        px_thresh_align = self.GAME_BIRD_PIXEL_RADIUS / 2 # Example threshold
        if bird_px_y_center < self.pipe_gap_y - px_thresh_align: pos_val = "below"
        elif bird_px_y_center > self.pipe_gap_y + px_thresh_align: pos_val = "above"
        else: pos_val = "aligned"
        
        dist_to_pipe_edge = self.pipe_x - 60 # Bird is at x=60
        if dist_to_pipe_edge > 150: dist_val = "far"
        elif dist_to_pipe_edge > 50: dist_val = "medium"
        else: dist_val = "close"

        if self.bird_velocity > 0.5: velo_val = "falling" # Adjusted threshold example
        elif self.bird_velocity < -0.5: velo_val = "rising"
        else: velo_val = "stable"
        
        # Create canonical (sorted) 3-part state string
        # This is the string policy keys should match if they are 3-part
        state_parts = [f"dist:{dist_val}", f"pos:{pos_val}", f"velo:{velo_val}"]
        return "_".join(sorted(state_parts))


OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")

def call_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=240)
        response.raise_for_status()
        response_text = response.json().get('response', '')
        json_start = response_text.find('[') if '[' in response_text else response_text.find('{')
        json_end = response_text.rfind(']') if '[' in response_text else response_text.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end+1]
            return json.loads(json_str)
        print(f"[Warning] LLM no valid JSON. Raw: {response_text[:200]}...")
        return None
    except requests.exceptions.RequestException as e: print(f"[Error] Ollama connection: {e}. Fallback."); return None
    except json.JSONDecodeError as e: print(f"[Error] JSON decode. Resp: {response_text[:200]}. Fallback."); return None


LLM_PROMPT_POLICY_FORMAT_GUIDANCE = """
Policy Format:
- A JSON object mapping state strings to action strings ('flap' or 'do_nothing').
- MUST include a "default" key (e.g., "default": "flap").
- State strings are composed of 1, 2, or 3 parts, joined by underscores.
- Each part MUST use the correct prefix:
    - 'pos:' for position (values: 'above', 'aligned', 'below')
    - 'dist:' for distance (values: 'far', 'medium', 'close')
    - 'velo:' for velocity (values: 'rising', 'stable', 'falling')
- For 2-part or 3-part state strings, the component parts MUST be alphabetically sorted before joining.
  Example 1-part: "pos:aligned"
  Example 2-part (sorted): "dist:medium_pos:aligned" (NOT "pos:aligned_dist:medium" if 'p' > 'd')
  Example 3-part (sorted): "dist:far_pos:above_velo:stable"
- More specific rules (3-part > 2-part > 1-part > default) take precedence during action selection.
"""

def llm_generate_initial_population(n=5, failed_policies=None):
    print(f"\n[LLM Call]: Generating initial population...")
    failed_policies = failed_policies or []
    failed_json = json.dumps(failed_policies[-3:], indent=2) if failed_policies else "[]" # Limit failed examples
    prompt = f"""
You are an expert AI game player for Flappy Bird. Design {n} diverse starting policies.
{LLM_PROMPT_POLICY_FORMAT_GUIDANCE}
Avoid policies similar to these recent failures: {failed_json}
Respond with ONLY the raw JSON array of {n} policies.
Example of a single policy (ensure parts are sorted for multi-part keys):
{{
  "dist:far_pos:below_velo:falling": "flap", 
  "dist:medium_pos:aligned": "flap", 
  "velo:stable": "do_nothing",
  "default": "do_nothing"
}}
"""
    generated_policies = call_ollama(prompt)
    if generated_policies and isinstance(generated_policies, list):
        normalized = [normalize_policy(p) for p in generated_policies[:n]]
        valid = [p for p in normalized if p and validate_policy(p)]
        if valid: print(f"[LLM Success] Generated {len(valid)} valid initial policies."); return valid
    print("[LLM Failure] Initial population. Using random.");
    return [generate_random_policy() for _ in range(n)]

def llm_intelligent_mutation(policy):
    print(f"[LLM Call]: Intelligent mutation...")
    policy_json = json.dumps(policy, indent=2)
    prompt = f"""
You are an AI policy optimizer for Flappy Bird. Mutate this policy to improve it:
{policy_json}
{LLM_PROMPT_POLICY_FORMAT_GUIDANCE}
Suggest a single, creative mutation (add/change/remove a rule, change default).
Return the complete new policy as a single JSON object. ONLY the JSON.
"""
    mutated_policy_data = call_ollama(prompt)
    mutated_policy = normalize_policy(mutated_policy_data)
    if mutated_policy and validate_policy(mutated_policy):
        print("[LLM Success] Valid mutation received."); return mutated_policy
    print("[LLM Failure] Mutation invalid. Falling back to random."); return None

def generate_random_policy():
    """Generates a simple random policy for fallback."""
    policy = {"default": random.choice(["flap", "do_nothing"])}
    # Add 0 to 2 random rules
    for _ in range(random.randint(0, 2)):
        num_parts = random.randint(1,3)
        key = generate_random_state_key(num_parts)
        if key not in policy:
            policy[key] = random.choice(["flap", "do_nothing"])
    return policy

def validate_policy(policy):
    if not isinstance(policy, dict): return False
    if "default" not in policy or policy["default"] not in ("flap", "do_nothing"): return False
    for key, value in policy.items():
        if value not in ("flap", "do_nothing"): return False
        if key != "default":
            parts = key.split('_')
            expected_prefixes = {"pos:", "dist:", "velo:"}
            actual_prefixes_in_key = set()
            
            for p_idx, p_val_str in enumerate(parts):
                valid_part = False
                for prefix in expected_prefixes:
                    if p_val_str.startswith(prefix):
                        actual_prefixes_in_key.add(prefix)
                        valid_part = True
                        # Basic value check (can be more specific)
                        val_part = p_val_str.split(":",1)[1]
                        if prefix == "pos:" and val_part not in ["above", "aligned", "below"]: valid_part=False; break
                        if prefix == "dist:" and val_part not in ["far", "medium", "close"]: valid_part=False; break
                        if prefix == "velo:" and val_part not in ["rising", "stable", "falling"]: valid_part=False; break
                        break
                if not valid_part:
                    # print(f"[Validate Fail] Invalid part '{p_val_str}' in key '{key}'")
                    return False
            
            # Check for duplicate prefixes (e.g. "pos:aligned_pos:below")
            if len(actual_prefixes_in_key) != len(parts):
                # print(f"[Validate Fail] Duplicate prefix types in key '{key}'")
                return False

            # Check if multi-part keys are sorted
            if len(parts) > 1:
                sorted_parts = sorted(parts)
                if parts != sorted_parts:
                    # print(f"[Validate Fail] Key not sorted: '{key}', should be '{'_'.join(sorted_parts)}'")
                    return False
    return True

def normalize_policy(policy_data): # Handles LLM output variations
    if isinstance(policy_data, list):
        if not policy_data: return None
        if len(policy_data) == 1 and isinstance(policy_data[0], dict):
            return normalize_policy(policy_data[0])
        result = {} # For list of state-action pairs (less common now)
        # ... (rest of list processing if needed) ...
        return result if result else None
    elif isinstance(policy_data, dict):
        if "default" in policy_data: return policy_data
        for key_name in ["policy", "mutated_policy", "strategy", "new_policy", "result"]:
            if key_name in policy_data and isinstance(policy_data[key_name], dict):
                return normalize_policy(policy_data[key_name])
        return policy_data # Assume it might be a policy missing 'default', validation will catch
    return None


def get_action_from_policy(policy, full_coded_state, action_map):
    # full_coded_state is assumed to be already sorted by get_coded_state()
    default_action_str = policy.get("default", "do_nothing")
    default_numeric_action = action_map.get(default_action_str, 0)

    # 1. Exact match for full 3-part state (which is already sorted)
    if full_coded_state in policy:
        return action_map.get(policy[full_coded_state], default_numeric_action)

    # 2. Parse the full_coded_state into its components
    #    Example: "dist:far_pos:aligned_velo:falling" (already sorted)
    current_state_parts = full_coded_state.split('_') # e.g., ["dist:far", "pos:aligned", "velo:falling"]
    
    # 3. Check for 2-part combinations
    #    Iterate through all 2-combinations of the current_state_parts
    if len(current_state_parts) == 3: # Only if full state gives 3 parts
        for i in range(3):
            for j in range(i + 1, 3):
                # Create a 2-part key from current_state_parts, already sorted by virtue of selection
                # and original full_coded_state being sorted.
                part1 = current_state_parts[i]
                part2 = current_state_parts[j]
                # Ensure part1 < part2 alphabetically for the key if not already guaranteed
                # (it is guaranteed if current_state_parts was sorted, which it is from get_coded_state)
                two_part_key = "_".join(sorted([part1, part2])) # Double sort to be safe
                if two_part_key in policy:
                    return action_map.get(policy[two_part_key], default_numeric_action)
    
    # 4. Check for 1-part combinations (which are just the elements of current_state_parts)
    for one_part_key in current_state_parts:
        if one_part_key in policy:
            return action_map.get(policy[one_part_key], default_numeric_action)
            
    return default_numeric_action


def evaluate_fitness(policy, game_sim):
    game_sim.reset()
    action_map = {"flap": 1, "do_nothing": 0}
    if not isinstance(policy, dict) or "default" not in policy: pass # get_action handles
    game_over = False
    while not game_over:
        current_coded_state = game_sim.get_coded_state()
        numeric_action = get_action_from_policy(policy, current_coded_state, action_map)
        game_over = game_sim.step(numeric_action)
        if game_sim.frames_survived > 4000: # Generous max frames
            break
    return (game_sim.frames_survived * 10) + (game_sim.score * 500)

def crossover(parent1, parent2):
    child_policy = {}
    keys = set(list(parent1.keys()) + list(parent2.keys()))
    for key in keys:
        if key == "default": continue
        child_policy[key] = parent1[key] if key in parent1 and random.random() < 0.5 else (parent2[key] if key in parent2 else None)
        if child_policy[key] is None: # If one parent didn't have the key, take from the other
            if key in parent1: child_policy[key] = parent1[key]
            elif key in parent2: child_policy[key] = parent2[key]
    child_policy["default"] = parent1.get("default", "do_nothing") if random.random() < 0.5 else parent2.get("default", "do_nothing")
    # Remove keys that ended up None (if neither parent had it but was in the combined set, though unlikely)
    return {k: v for k, v in child_policy.items() if v is not None}


def generate_random_state_key(num_parts):
    """ Helper to generate a sorted, valid random state key of num_parts components."""
    pos_vals = ["above", "aligned", "below"]
    dist_vals = ["far", "medium", "close"]
    velo_vals = ["rising", "stable", "falling"]
    
    components = []
    # Ensure unique component types if num_parts < 3
    available_types = [("pos", pos_vals), ("dist", dist_vals), ("velo", velo_vals)]
    random.shuffle(available_types) # So we don't always pick pos then dist etc.

    for i in range(num_parts):
        prefix_str, val_list = available_types[i]
        components.append(f"{prefix_str}:{random.choice(val_list)}")
        
    return "_".join(sorted(components))


def mutate(policy, use_llm=False):
    mutated_policy_llm = None
    if use_llm and random.random() < 0.33:
        mutated_policy_llm = llm_intelligent_mutation(policy.copy())
    if mutated_policy_llm and validate_policy(mutated_policy_llm):
        return mutated_policy_llm
    
    new_policy = policy.copy()
    mutation_type = random.random()

    if mutation_type < 0.1 and len(new_policy) > 1:
        keys = [k for k in new_policy.keys() if k != "default"]
        if keys: del new_policy[random.choice(keys)]
    elif mutation_type < 0.5:
        key_to_mutate = random.choice(list(new_policy.keys()))
        new_policy[key_to_mutate] = "flap" if new_policy[key_to_mutate] == "do_nothing" else "do_nothing"
    else:
        num_parts_for_new_rule = random.randint(1, 3)
        new_state_key = generate_random_state_key(num_parts_for_new_rule)
        if new_state_key not in new_policy:
            new_policy[new_state_key] = random.choice(["flap", "do_nothing"])
        elif "default" in new_policy: # Fallback: change default if key exists
             new_policy["default"] = "flap" if new_policy["default"] == "do_nothing" else "do_nothing"

    if "default" not in new_policy:
        new_policy["default"] = random.choice(["flap", "do_nothing"])
    return new_policy


_shared_game_eval_instance = FlappyBirdGame() # For headless evaluation

def run_flappy_game(policy=None, interactive=False, max_frames=10000, render=True, return_fitness=False):
    # (This function remains largely the same as before, using get_action_from_policy)
    # Minor adjustments for clarity if needed, but the core logic for action selection
    # via get_action_from_policy is the key change already detailed.
    game_instance = FlappyBirdGame()
    if render:
        if not pygame.get_init(): pygame.init()
        if not pygame.font.get_init(): pygame.font.init()
        WIDTH, HEIGHT = 400, FlappyBirdGame.GAME_SCREEN_HEIGHT
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy Evo")
        try: font = pygame.font.SysFont(None, 32)
        except pygame.error: font = pygame.font.Font(None, 32) 
    else: screen = None

    clock = pygame.time.Clock()
    action_map = {"flap": 1, "do_nothing": 0}
    running = True
    game_over = False

    while running and not game_over:
        action_to_take = 0 
        if interactive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; game_over = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: action_to_take = 1
            if not running: break
        elif policy:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False; game_over = True
            if not running: break
            state_code = game_instance.get_coded_state()
            action_to_take = get_action_from_policy(policy, state_code, action_map)
        else:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False; game_over = True
            if not running: break
            action_to_take = 0 

        if not game_over:
            game_over = game_instance.step(action_to_take)

        if render and screen:
            screen.fill((135, 206, 235))
            pipe_color = (34, 139, 34)
            uig_h = game_instance.pipe_gap_y - game_instance.gap_height // 2
            lig_y = game_instance.pipe_gap_y + game_instance.gap_height // 2
            pygame.draw.rect(screen, pipe_color, (game_instance.pipe_x, 0, game_instance.pipe_width, uig_h))
            pygame.draw.rect(screen, pipe_color, (game_instance.pipe_x, lig_y, game_instance.pipe_width, FlappyBirdGame.GAME_SCREEN_HEIGHT - lig_y))
            bird_color = (255, 215, 0)
            bird_rx = 60
            bird_ryc = int((game_instance.bird_y / FlappyBirdGame.BIRD_Y_MAX) * FlappyBirdGame.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE)
            pygame.draw.circle(screen, bird_color, (bird_rx, bird_ryc), FlappyBirdGame.GAME_BIRD_PIXEL_RADIUS)
            if policy or interactive :
                score_surf = font.render(f"Score: {game_instance.score}", True, (0,0,0))
                screen.blit(score_surf, (10, 10))
                frames_surf = font.render(f"Frames: {game_instance.frames_survived}", True, (0,0,0))
                screen.blit(frames_surf, (10, 40))
            pygame.display.flip()
            clock.tick(60)

        if game_instance.frames_survived >= max_frames: game_over = True

        if game_over and running:
            if render and interactive:
                font_big = pygame.font.SysFont(None, 48)
                msg_text = f"Game Over! Score: {game_instance.score}"
                msg = font_big.render(msg_text, True, (255,0,0))
                msg_rect = msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
                screen.blit(msg, msg_rect)
                prompt_text = "Press any key to return to menu..."
                prompt_surf = font.render(prompt_text, True, (200,200,200))
                prompt_rect = prompt_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
                screen.blit(prompt_surf, prompt_rect)
                pygame.display.flip()
                waiting = True
                while waiting and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: waiting=False; running=False
                        if event.type == pygame.KEYDOWN: waiting=False
                    clock.tick(15)
            elif render : pygame.time.wait(200) # Short pause for AI crash
            running = False 

    if return_fitness:
        return (game_instance.frames_survived * 10) + (game_instance.score * 500)
    return None

def play_game_with_policy(policy_to_play=None):
    # ... (same as before) ...
    loaded_policy = policy_to_play
    if loaded_policy is None:
        try:
            with open("best_policy.json", "r") as f: loaded_policy = json.load(f)
            print("[INFO] Loaded best_policy.json for playback.")
        except Exception as e:
            print(f"[ERROR] No best_policy.json: {e}. Train first.")
            return
    if not validate_policy(loaded_policy): # Validate before playing
        print(f"[ERROR] Loaded policy invalid. Cannot play. Policy: {loaded_policy}")
        return
    run_flappy_game(policy=loaded_policy, interactive=False, render=True, max_frames=20000)


def play_game_interactive():
    # ... (same as before) ...
    run_flappy_game(policy=None, interactive=True, render=True, max_frames=60000)

def main_menu():
    # ... (same as before, ensure get_menu_items_list calls validate_policy) ...
    pygame.init()
    WIDTH, HEIGHT = 400, FlappyBirdGame.GAME_SCREEN_HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Evo - Main Menu")
    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    current_best_policy = None
    policy_file_checked_this_session = False
    policy_file_is_valid = False

    def get_menu_items_list():
        nonlocal current_best_policy, policy_file_checked_this_session, policy_file_is_valid
        items = []
        if current_best_policy is None and not policy_file_checked_this_session:
            try:
                with open("best_policy.json", "r") as f:
                    loaded_p = json.load(f)
                    if validate_policy(loaded_p): # Crucial validation
                        current_best_policy = loaded_p
                        policy_file_is_valid = True
                        print("[INFO] Menu: Loaded valid best_policy.json.")
                    else:
                        print("[INFO] Menu: best_policy.json found but invalid according to new rules.")
                        policy_file_is_valid = False
            except FileNotFoundError: # ... (rest of error handling) ...
                print("[INFO] Menu: best_policy.json not found.")
                policy_file_is_valid = False
            except json.JSONDecodeError:
                print("[INFO] Menu: best_policy.json corrupted.")
                policy_file_is_valid = False
            except Exception as e:
                print(f"[INFO] Menu: Error loading best_policy.json: {e}")
                policy_file_is_valid = False
            policy_file_checked_this_session = True
        
        if current_best_policy and policy_file_is_valid:
            items.append(("Play Best Policy", "play_best"))
        items.extend([
            ("Train New Policy (Evolution)", "train"),
            ("Play Yourself", "play_human"),
            ("Quit", "quit")])
        return items

    selected_index = 0
    menu_running = True
    while menu_running: # ... (rest of main_menu loop as before, using validate_policy on newly trained policies) ...
        if not pygame.display.get_init() or pygame.display.get_surface() is None:
            screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
            pygame.display.set_caption("Flappy Evo - Main Menu")
        
        menu_options_list = get_menu_items_list()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: menu_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: selected_index = (selected_index - 1 + len(menu_options_list)) % len(menu_options_list) if menu_options_list else 0
                elif event.key == pygame.K_DOWN: selected_index = (selected_index + 1) % len(menu_options_list) if menu_options_list else 0
                elif event.key == pygame.K_RETURN and menu_options_list:
                    action_key = menu_options_list[selected_index][1]
                    if action_key == "play_best":
                        if current_best_policy and policy_file_is_valid: play_game_with_policy(current_best_policy)
                        else: print("[Menu] No valid best policy loaded to play.")
                    elif action_key == "train":
                        new_policy = train_policy() # This returns the best policy from training
                        if new_policy and validate_policy(new_policy):
                            current_best_policy = new_policy
                            policy_file_is_valid = True # Mark newly trained policy as valid
                        else: # If training didn't yield a valid policy, clear current
                            current_best_policy = None
                            policy_file_is_valid = False
                        policy_file_checked_this_session = False # Allow re-check of file on next menu draw
                    elif action_key == "play_human": play_game_interactive()
                    elif action_key == "quit": menu_running = False
                    
                    # Refresh menu options and ensure selected_index is valid
                    menu_options_list = get_menu_items_list() 
                    selected_index = min(selected_index, len(menu_options_list) -1) if menu_options_list else 0


        screen.fill((30, 30, 30))
        title_surf = font_large.render("Flappy Evo", True, (255,255,255))
        screen.blit(title_surf, title_surf.get_rect(center=(WIDTH // 2, 80)))
        for i, (text, _) in enumerate(menu_options_list):
            color = (255,255,0) if i == selected_index else (200,200,200)
            item_surf = font_small.render(text, True, color)
            screen.blit(item_surf, item_surf.get_rect(center=(WIDTH // 2, 200 + i * 60)))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def train_policy():
    # ... (train_policy setup, logging, Pygame plot as before) ...
    # Key change is to ensure policies generated (randomly or by LLM)
    # and mutated adhere to the new sorted key format.
    # The llm_generate_initial_population and mutate functions now incorporate this.
    NUM_GENERATIONS = 100
    POPULATION_SIZE = 20
    NUM_PARENTS = 5

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"training_log_{timestamp}.csv"
    log_file_path = os.path.join(os.getcwd(), log_filename)
    try:
        with open(log_file_path, 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(['Generation', 'Policy_Index', 'Fitness', 'Policy_JSON'])
        print(f"[INFO] Training log: {log_file_path}")
    except IOError as e:
        print(f"[ERROR] Log file creation failed: {e}. Logging disabled.")
        log_file_path = None

    pygame.init()
    PLOT_WIDTH, PLOT_HEIGHT = 800, 600
    plot_screen = pygame.display.set_mode((PLOT_WIDTH, PLOT_HEIGHT))
    pygame.display.set_caption("Flappy Evo - Training Progress")
    font_plot = pygame.font.SysFont(None, 24)
    font_plot_small = pygame.font.SysFont(None, 18)
    # ... (plot layout variables as before) ...
    plot_margin = 60
    graph_width = PLOT_WIDTH - 2 * plot_margin
    graph_height = PLOT_HEIGHT // 2 - plot_margin - 40
    
    fitness_history = []
    overall_best_policy = None
    overall_best_fitness = -float('inf')

    failed_policies_for_llm = []
    population = llm_generate_initial_population(POPULATION_SIZE, failed_policies=failed_policies_for_llm)
    # Ensure all initial policies are valid after generation
    population = [p for p in population if p and validate_policy(p)] 
    while len(population) < POPULATION_SIZE:
        population.append(generate_random_policy()) # Use helper for valid random policies

    print(f"\n--- Training: {NUM_GENERATIONS} gens, {POPULATION_SIZE} pop ---")

    for gen in range(NUM_GENERATIONS): # ... (rest of training loop as before, using new helper functions) ...
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit() 
                print("Training aborted.")
                if overall_best_policy: # Save best if exists
                    try:
                        with open("best_policy.json", "w") as f: json.dump(overall_best_policy, f, indent=2)
                        print(f"Saved best policy from abort (Fitness: {overall_best_fitness}).")
                    except Exception as e: print(f"Error saving policy on abort: {e}")
                return overall_best_policy # Return what we have
        
        fitness_scores = [evaluate_fitness(p, _shared_game_eval_instance) for p in population]
        
        if log_file_path: # Log to CSV
            try:
                with open(log_file_path, 'a', newline='') as csvfile:
                    log_writer = csv.writer(csvfile)
                    for i, (p_item, f_s) in enumerate(zip(population, fitness_scores)):
                        log_writer.writerow([gen + 1, i, f_s, json.dumps(p_item)])
            except IOError as e:
                print(f"[ERROR] Log write failed: {e}. Logging disabled for session.")
                log_file_path = None

        pop_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        current_gen_best_p, current_gen_best_f = pop_with_fitness[0]

        if current_gen_best_f > overall_best_fitness:
            overall_best_fitness = current_gen_best_f
            overall_best_policy = current_gen_best_p.copy() # Store a copy
            print(f"Gen {gen+1}: New Overall Best Fitness: {overall_best_fitness:.0f}")

        fitness_history.append(current_gen_best_f)

        # Update failed policies for LLM (example: bottom 20% or very low absolute fitness)
        # For simplicity, let's keep the previous threshold method for now
        for p_item, score_val in pop_with_fitness:
            if score_val < 1000 and p_item not in failed_policies_for_llm: # Adjusted threshold for new game difficulty
                 failed_policies_for_llm.append(p_item)
        if len(failed_policies_for_llm) > 10: failed_policies_for_llm = failed_policies_for_llm[-10:]

        parents = [p_data[0] for p_data in pop_with_fitness[:NUM_PARENTS]]
        next_generation = parents[:] # Elitism

        while len(next_generation) < POPULATION_SIZE:
            p1 = random.choice(parents) if parents else generate_random_policy()
            p2 = random.choice(parents) if len(parents) > 1 else generate_random_policy()
            child = crossover(p1, p2)
            child = mutate(child, use_llm=True) # LLM mutation enabled
            if validate_policy(child): 
                next_generation.append(child)
            else: 
                # print(f"[Warning] Invalid child in Gen {gen+1}. Using random fallback.")
                next_generation.append(generate_random_policy()) # Ensure valid policy added
        population = next_generation[:POPULATION_SIZE]


        # --- Plotting ---
        plot_screen.fill((30, 30, 30))
        pygame.draw.line(plot_screen, (200,200,200), (plot_margin, plot_margin), (plot_margin, plot_margin + graph_height), 2)
        pygame.draw.line(plot_screen, (200,200,200), (plot_margin, plot_margin + graph_height), (plot_margin + graph_width, plot_margin + graph_height), 2)

        if len(fitness_history) > 1:
            max_f_hist = max(fitness_history) if fitness_history else 1.0 # Avoid zero division
            min_f_hist = min(fitness_history) if fitness_history else 0.0
            f_range = max_f_hist - min_f_hist
            if f_range <= 0: f_range = 1.0 # Avoid zero division if all values are same

            points = []
            for i_fh, f_val in enumerate(fitness_history):
                x_coord = plot_margin + int((i_fh / max(1, NUM_GENERATIONS -1)) * graph_width)
                y_coord = plot_margin + graph_height - int(((f_val - min_f_hist) / f_range) * graph_height)
                points.append((x_coord,y_coord))
            if len(points)>1: pygame.draw.lines(plot_screen, (255,215,0), False, points, 2)

        label_txt = f"Gen: {gen+1}/{NUM_GENERATIONS} | Overall Best: {overall_best_fitness:.0f} | Cur Gen Best: {current_gen_best_f:.0f}"
        label_s = font_plot.render(label_txt, True, (255,255,255))
        plot_screen.blit(label_s, (plot_margin, plot_margin + graph_height + 10))
        
        if overall_best_policy: # Display current best policy
            policy_disp_y = plot_margin + graph_height + 40
            try:
                policy_lines = json.dumps(overall_best_policy, indent=1).splitlines()
                for i, line in enumerate(policy_lines):
                    if policy_disp_y + i * 18 < PLOT_HEIGHT -10: # Check bounds
                        line_s = font_plot_small.render(line, True, (200,255,200))
                        plot_screen.blit(line_s, (plot_margin, policy_disp_y + i * 18))
                    else: break
            except Exception: pass # Ignore policy display errors

        pygame.display.flip()
        # pygame.time.delay(10) # Optional delay

    print("\n--- Evolution Complete ---")
    if overall_best_policy:
        print("Final Best Policy:")
        print(json.dumps(overall_best_policy, indent=2))
        try:
            with open("best_policy.json", "w") as f: json.dump(overall_best_policy, f, indent=2)
            print("Saved to best_policy.json")
        except Exception as e: print(f"Error saving best_policy.json: {e}")
    else: print("No valid best policy was found during training.")
    
    # pygame.display.quit() # Quit only the plot display, not all of pygame if menu is to be shown again
    # The main_menu will handle final pygame.quit()
    return overall_best_policy


if __name__ == "__main__":
    main_menu()