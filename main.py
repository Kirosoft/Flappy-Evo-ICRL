import os
from dotenv import load_dotenv
import random
import json
import requests
import pygame
import csv
import time
import threading
import shutil

load_dotenv()

# --- Load Prompt File Paths from .env ---
POLICY_FORMAT_GUIDANCE_PATH = os.getenv("POLICY_FORMAT_GUIDANCE_PATH", "prompts/policy_format_guidance.md")
EXAMPLE_POLICY_PATH = os.getenv("EXAMPLE_POLICY_PATH", "prompts/example_policy.json")
INITIAL_POP_TEMPLATE_PATH = os.getenv("INITIAL_POP_TEMPLATE_PATH", "prompts/initial_population_template.md")
MUTATION_TEMPLATE_PATH = os.getenv("MUTATION_TEMPLATE_PATH", "prompts/intelligent_mutation_template.md")

def read_file_content(file_path, default_content=""):
    try:
        with open(file_path, "r", encoding="utf-8") as f: return f.read()
    except FileNotFoundError: print(f"[Warning] Prompt file not found: {file_path}."); return default_content
    except Exception as e: print(f"[Error] Reading prompt file {file_path}: {e}."); return default_content

LLM_PROMPT_POLICY_FORMAT_GUIDANCE = read_file_content(POLICY_FORMAT_GUIDANCE_PATH, "Error: Policy format guidance missing.")
LLM_EXAMPLE_POLICY_JSON_STR = read_file_content(EXAMPLE_POLICY_PATH, '{ "default": "flap" }')
LLM_INITIAL_POP_TEMPLATE = read_file_content(INITIAL_POP_TEMPLATE_PATH, "Error: Initial population template missing.")
LLM_MUTATION_TEMPLATE = read_file_content(MUTATION_TEMPLATE_PATH, "Error: Mutation template missing.")

# --- Gameplay Settings & LLM Settings (as before) ---
PIPE_START_X_SETTING = 500; PIPE_SPEED_SETTING = 3; PIPE_RESPAWN_X_SETTING = 300 # ... etc.
PIPE_GAP_MIN_SETTING = 130; PIPE_GAP_MAX_SETTING = 170; BIRD_Y_MIN_SETTING = 0
BIRD_Y_MAX_SETTING = 100; GRAVITY_SETTING = 0.10; FLAP_STRENGTH_SETTING = -1.50
PIPE_MARGIN_Y_SETTING = 50
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:!4b")
LLM_TIMEOUT = 240; LLM_TEMPERATURE_INIT_POP = 0.7; LLM_TEMPERATURE_MUTATION = 0.4
FAILED_LLM_LOG_FILE = "failed_llm_prompts_log.txt"
PER_GENERATION_BEST_DIR = "generation_best_policies"

def log_failed_llm_attempt(prompt, raw_response, error_type="Unknown Validation/Parse Error"):
    # ... (same as before) ...
    try:
        with open(FAILED_LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Failed LLM Attempt: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Error Type: {error_type}\n"); f.write("Prompt Sent:\n"); f.write(prompt + "\n\n")
            f.write("Raw LLM Response Received:\n"); f.write(str(raw_response) + "\n")
            f.write("------------------------------------------------------------\n\n")
    except Exception as e: print(f"[ERROR] Could not write to failed LLM prompt log: {e}")

class FlappyBirdGame: # (Same as before)
    PIPE_START_X = PIPE_START_X_SETTING; PIPE_SPEED = PIPE_SPEED_SETTING; PIPE_RESPAWN_X = PIPE_RESPAWN_X_SETTING
    PIPE_GAP_MIN = PIPE_GAP_MIN_SETTING; PIPE_GAP_MAX = PIPE_GAP_MAX_SETTING; BIRD_Y_MIN = BIRD_Y_MIN_SETTING
    BIRD_Y_MAX = BIRD_Y_MAX_SETTING; GRAVITY = GRAVITY_SETTING; FLAP_STRENGTH = FLAP_STRENGTH_SETTING
    PIPE_MARGIN_Y = PIPE_MARGIN_Y_SETTING; GAME_SCREEN_HEIGHT = 600; GAME_BIRD_PIXEL_RADIUS = 20
    GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE = GAME_SCREEN_HEIGHT-(2*GAME_BIRD_PIXEL_RADIUS)
    def __init__(self): # ... (constructor) ...
        self.bird_y=50; self.bird_velocity=0; self.gravity=self.GRAVITY; self.flap_strength=self.FLAP_STRENGTH
        self.pipe_x=self.PIPE_START_X; self.score=0; self.frames_survived=0; self.pipe_width=60
        self.gap_height=random.randint(self.PIPE_GAP_MIN,self.PIPE_GAP_MAX); self.pipe_gap_y=self._random_gap_y(self.gap_height)
    def _random_gap_y(self, gap_height_pixels): # ... (method) ...
        min_center_y=self.PIPE_MARGIN_Y+gap_height_pixels//2; max_center_y=self.GAME_SCREEN_HEIGHT-self.PIPE_MARGIN_Y-gap_height_pixels//2
        return random.randint(min_center_y,max_center_y) if min_center_y < max_center_y else self.GAME_SCREEN_HEIGHT//2
    def reset(self): self.__init__()
    def step(self, action): # ... (method) ...
        self.frames_survived+=1
        if action==1: self.bird_velocity=self.flap_strength
        self.bird_velocity+=self.gravity; self.bird_y+=self.bird_velocity
        self.bird_y=max(self.BIRD_Y_MIN,min(self.BIRD_Y_MAX,self.bird_y)); self.pipe_x-=self.PIPE_SPEED
        if self.pipe_x+self.pipe_width < 0:
            self.pipe_x=self.PIPE_RESPAWN_X; self.gap_height=random.randint(self.PIPE_GAP_MIN,self.PIPE_GAP_MAX)
            self.pipe_gap_y=self._random_gap_y(self.gap_height); self.score+=1
        is_collision=False; bird_px_x_center=60; bird_px_radius=self.GAME_BIRD_PIXEL_RADIUS
        bird_px_left=bird_px_x_center-bird_px_radius; bird_px_right=bird_px_x_center+bird_px_radius
        bird_px_y_center=(self.bird_y/self.BIRD_Y_MAX)*self.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE
        bird_px_top=bird_px_y_center-bird_px_radius; bird_px_bottom=bird_px_y_center+bird_px_radius
        pipe_px_left=self.pipe_x; pipe_px_right=self.pipe_x+self.pipe_width
        gap_px_top=self.pipe_gap_y-self.gap_height//2; gap_px_bottom=self.pipe_gap_y+self.gap_height//2
        if bird_px_right > pipe_px_left and bird_px_left < pipe_px_right:
            if bird_px_top < gap_px_top or bird_px_bottom > gap_px_bottom: is_collision=True
        if self.bird_y >= self.BIRD_Y_MAX or self.bird_y <= self.BIRD_Y_MIN: is_collision=True
        return is_collision
    def get_coded_state(self): # ... (method - already returns sorted 3-part key) ...
        bird_px_y_center=(self.bird_y/self.BIRD_Y_MAX)*self.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE
        px_thresh_align=self.GAME_BIRD_PIXEL_RADIUS/2
        if bird_px_y_center < self.pipe_gap_y-px_thresh_align: pos_val="below"
        elif bird_px_y_center > self.pipe_gap_y+px_thresh_align: pos_val="above"
        else: pos_val="aligned"
        dist_to_pipe_edge=self.pipe_x-60
        if dist_to_pipe_edge > 150: dist_val="far"
        elif dist_to_pipe_edge > 50: dist_val="medium"
        else: dist_val="close"
        if self.bird_velocity > 0.5: velo_val="falling"
        elif self.bird_velocity < -0.5: velo_val="rising"
        else: velo_val="stable"
        state_parts=[f"dist:{dist_val}",f"pos:{pos_val}",f"velo:{velo_val}"]
        return "_".join(sorted(state_parts))


def call_ollama_blocking(prompt, temperature): # (Same as before)
    # ...
    raw_response_text_for_log = "No response captured before error." 
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=LLM_TIMEOUT)
        raw_response_text_for_log = response.text 
        response.raise_for_status()
        response_dict = response.json(); 
        json_string_raw = response_dict.get('response', '')
        #json_string = json_string_raw.split("/think>")[1].strip() # ok this needs to be better
        json_string = json_string_raw.split("```json")[1].strip() # ok this needs to be better
        json_string = json_string.split("```")[0] # ok this needs to be better
        if not json_string:
            log_failed_llm_attempt(prompt, raw_response_text_for_log, "LLM empty 'response' field")
            return None, raw_response_text_for_log
        parsed_json = json.loads(json_string)
        return parsed_json, json_string
    except requests.exceptions.Timeout:
        log_failed_llm_attempt(prompt, None, f"Timeout {LLM_TIMEOUT}s"); return None, "Request Timed Out"
    except requests.exceptions.RequestException as e:
        log_failed_llm_attempt(prompt,raw_response_text_for_log,f"Ollama connection error: {e}"); return None, f"Ollama Connection Error: {raw_response_text_for_log}"
    except json.JSONDecodeError as e:
        log_failed_llm_attempt(prompt,json_string if 'json_string' in locals() else raw_response_text_for_log,f"JSON decode error: {e}"); return None, json_string if 'json_string' in locals() else raw_response_text_for_log
    except Exception as e:
        log_failed_llm_attempt(prompt,raw_response_text_for_log,f"Unexpected error: {e}"); return None, f"Unexpected Error: {raw_response_text_for_log}"


def call_ollama_blocking_for_thread(prompt, temperature, result_container, error_container, raw_response_container): # (Same as before)
    # ...
    parsed_data, raw_text = call_ollama_blocking(prompt, temperature)
    raw_response_container['text'] = raw_text
    if parsed_data is not None: result_container['data'] = parsed_data
    else: error_container['error'] = f"LLM call failed. Raw: {(raw_text[:200] + '...') if raw_text else 'N/A'}"


def llm_generate_initial_population_prompt_str(n=5, failed_policies=None): # (Same as before)
    # ...
    failed_policies = failed_policies or []
    failed_json_str = json.dumps(failed_policies[-2:], indent=2) if failed_policies else "[]"
    prompt = LLM_INITIAL_POP_TEMPLATE.format(n=n, policy_format_guidance=LLM_PROMPT_POLICY_FORMAT_GUIDANCE,
        failed_json_str=failed_json_str, example_policy_json_str=LLM_EXAMPLE_POLICY_JSON_STR)
    return prompt

def llm_intelligent_mutation(policy): # (Same as before, uses normalize_policy)
    # ...
    print(f"[LLM Call - Mutate]: Asking '{OLLAMA_MODEL}'...")
    policy_to_mutate_json_str = json.dumps(policy, indent=2)
    prompt = LLM_MUTATION_TEMPLATE.format(policy_to_mutate_json_str=policy_to_mutate_json_str,
        policy_format_guidance=LLM_PROMPT_POLICY_FORMAT_GUIDANCE, example_policy_json_str=LLM_EXAMPLE_POLICY_JSON_STR)
    parsed_data, raw_response = call_ollama_blocking(prompt, LLM_TEMPERATURE_MUTATION)
    if parsed_data is None: print(f"[LLM Failure - Mutate] Invalid/No response. See '{FAILED_LLM_LOG_FILE}'."); return None
    mutated_policy = normalize_policy(parsed_data) # <<< THIS IS WHERE THE FIX GOES
    if mutated_policy and validate_policy(mutated_policy):
        print("[LLM Success - Mutate] Valid mutation."); return mutated_policy
    else:
        print(f"[LLM Failure - Mutate] Validation fail. Norm: {mutated_policy}. See '{FAILED_LLM_LOG_FILE}'.")
        log_failed_llm_attempt(prompt, raw_response or "No raw response", f"Validation failed. Norm: {mutated_policy}")
        return None

def generate_random_policy(): # (Same as before)
    # ...
    policy = {"default": random.choice(["flap", "do_nothing"])}
    for _ in range(random.randint(0, 2)):
        key = generate_random_state_key(random.randint(1,3))
        if key and key not in policy: policy[key] = random.choice(["flap", "do_nothing"])
    return policy


# ***************************************************************************
# * THE CRUCIAL CHANGE IS IN normalize_policy and validate_policy           *
# ***************************************************************************

# ... (other functions and imports as before) ...

def _normalize_and_correct_single_policy_dict(llm_policy_dict_input):
    """
    Processes a single dictionary that is supposed to be a policy.
    - Handles common LLM wrapping (e.g., "policy": {...}).
    - Ensures a valid "default" action exists.
    - Corrects simple action typos (e.g., "do_no'thing").
    - Sorts components of multi-part state keys alphabetically.
    - Discards keys that are clearly not valid state definitions.
    Returns a processed policy dictionary, or None if input is not a dict.
    """
    if not isinstance(llm_policy_dict_input, dict):
        # print(f"[Normalize DEBUG] Input to _normalize_and_correct_single_policy_dict is not a dict: {type(llm_policy_dict_input)}")
        return None

    # 1. Attempt to unwrap if nested (e.g., {"policy": {...}})
    current_dict_to_process = llm_policy_dict_input
    if "default" not in current_dict_to_process: # A common sign it might be wrapped
        for wrapper_key in ["policy", "mutated_policy", "strategy", "new_policy", "result"]:
            if wrapper_key in current_dict_to_process and isinstance(current_dict_to_process[wrapper_key], dict):
                # print(f"[Normalize DEBUG] Unwrapped policy from key '{wrapper_key}'")
                current_dict_to_process = current_dict_to_process[wrapper_key]
                break 
    
    if not isinstance(current_dict_to_process, dict): # Check again after potential unwrap
         # print(f"[Normalize DEBUG] Not a dict after unwrap attempt.")
         return None

    # 2. Initialize the new policy we are building
    normalized_policy = {}

    # 3. Handle the "default" key specifically
    original_default_action = current_dict_to_process.get("default")
    if original_default_action == "do_no'thing": # Specific typo fix
        normalized_policy["default"] = "do_nothing"
        # print(f"[Normalize DEBUG] Corrected 'do_no'thing' typo in default action.")
    elif original_default_action in ("flap", "do_nothing"):
        normalized_policy["default"] = original_default_action
    else:
        # if original_default_action is not None:
        #     print(f"[Normalize WARN] Invalid or missing default action '{original_default_action}'. Assigning random.")
        #     log_failed_llm_attempt(str(llm_policy_dict_input), str(current_dict_to_process), 
        #                            f"Invalid/Missing default action: {original_default_action}")
        normalized_policy["default"] = random.choice(["flap", "do_nothing"])

    # 4. Process all other keys (potential state rules)
    for key, value in current_dict_to_process.items():
        if key == "default":
            continue # Already handled

        # Correct simple action typos for the current rule's value
        current_action_value = value
        if isinstance(value, str) and value == "do_no'thing":
            current_action_value = "do_nothing"
            # print(f"[Normalize DEBUG] Corrected 'do_no'thing' typo for key '{key}'")
        
        # Validate the action value before proceeding with key processing
        if current_action_value not in ("flap", "do_nothing"):
            # print(f"[Normalize WARN] Discarding rule with invalid action value '{value}' for key '{key}'")
            # log_failed_llm_attempt(str(llm_policy_dict_input), str(current_dict_to_process), 
            #                        f"Discarded rule with invalid action: {key}: {value}")
            continue # Skip this rule

        # Process the key itself
        parts = key.split('_')
        if not parts: # Should not happen if key is not empty
            # print(f"[Normalize WARN] Discarding empty key string for value '{current_action_value}'")
            continue

        # Check if all parts of the key have a valid prefix
        is_key_structurally_valid = True
        component_prefixes = {"pos:", "dist:", "velo:"}
        for part_str in parts:
            has_valid_prefix_for_part = False
            for prefix in component_prefixes:
                if part_str.startswith(prefix):
                    has_valid_prefix_for_part = True
                    break
            if not has_valid_prefix_for_part:
                is_key_structurally_valid = False
                break
        
        if not is_key_structurally_valid:
            # print(f"[Normalize WARN] Discarding key '{key}' due to invalid part structure/prefix.")
            # log_failed_llm_attempt(str(llm_policy_dict_input), str(current_dict_to_process),
            #                        f"Discarded key with invalid part structure: {key}")
            continue # Skip this key

        # If key structure is plausible, sort components for multi-part keys
        if len(parts) > 1:
            final_key = "_".join(sorted(parts))
        else: # Single part key
            final_key = parts[0] 
        
        normalized_policy[final_key] = current_action_value
        
    return normalized_policy


import random
import json # For potential use in debugging, not strictly required by the function itself

# Assume log_failed_llm_attempt is defined elsewhere if you want to use it for logging errors
# def log_failed_llm_attempt(prompt_info, raw_response, error_type):
#     print(f"[LOG_FAIL] Error: {error_type}, Raw: {raw_response}, Prompt: {prompt_info}")


def normalize_policy(policy_data_input, expect_list=False):
    """
    Normalizes LLM policy output.
    If expect_list is True:
        - If a single dict is given, it's wrapped in a list.
        - Each item in the (potentially new) list is then processed as a policy dict.
        - Returns a list of processed policy dicts, or an empty list if all items failed.
    If expect_list is False:
        - Expects a single policy dict (possibly wrapped in a list of one).
        - Processes it as a single policy dict.
        - Returns a single processed policy dict, or None if processing failed.
    Returns None if the input structure is fundamentally incompatible before processing.
    """

    # --- Helper to process a single potential policy dictionary ---
    def _normalize_and_correct_single_policy_dict(llm_policy_dict_item):
        if not isinstance(llm_policy_dict_item, dict):
            # print(f"[Normalize DEBUG] _normalize_single: Input not a dict: {type(llm_policy_dict_item)}")
            return None 

        current_dict_to_process = llm_policy_dict_item
        # Attempt to unwrap if it looks like { "wrapper_key": { actual_policy_content } }
        if "default" not in current_dict_to_process: 
            for wrapper_key in ["policy", "mutated_policy", "strategy", "new_policy", "result"]:
                if wrapper_key in current_dict_to_process and isinstance(current_dict_to_process[wrapper_key], dict):
                    # print(f"[Normalize DEBUG] Unwrapped policy from key '{wrapper_key}'")
                    current_dict_to_process = current_dict_to_process[wrapper_key]
                    break 
        
        if not isinstance(current_dict_to_process, dict):
             # print(f"[Normalize DEBUG] _normalize_single: Not a dict after unwrap attempt.")
             return None

        normalized_policy = {} # Initialize the new policy we are building

        # 1. Handle the "default" key specifically
        original_default_action = current_dict_to_process.get("default")
        if isinstance(original_default_action, str) and original_default_action.lower() == "do_no'thing": # Specific typo fix
            normalized_policy["default"] = "do_nothing"
            # print(f"[Normalize DEBUG] Corrected 'do_no'thing' typo in default action for input: {str(llm_policy_dict_item)[:100]}")
        elif original_default_action in ("flap", "do_nothing"):
            normalized_policy["default"] = original_default_action
        else:
            # if original_default_action is not None:
            #     print(f"[Normalize WARN] Invalid or missing default action '{original_default_action}' in {str(llm_policy_dict_item)[:100]}. Assigning random.")
            #     # log_failed_llm_attempt(str(llm_policy_dict_item), str(current_dict_to_process), 
            #     #                        f"Invalid/Missing default action: {original_default_action}")
            # else:
            #      print(f"[Normalize WARN] Missing 'default' key in {str(llm_policy_dict_item)[:100]}. Assigning random.")
            #      # log_failed_llm_attempt(str(llm_policy_dict_item), str(current_dict_to_process), "Missing default key")
            normalized_policy["default"] = random.choice(["flap", "do_nothing"])

        # 2. Process all other keys (potential state rules)
        for key, value in current_dict_to_process.items():
            # print(f"[Normalize DEBUG] _normalize_single: Processing key: {repr(key)}, value: {repr(value)}")
            if key == "default":
                continue # Already handled

            # Correct simple action typos for the current rule's value
            current_action_value = value
            if isinstance(value, str) and value.lower() == "do_no'thing": # case-insensitive for typo
                current_action_value = "do_nothing"
                # print(f"[Normalize DEBUG] Corrected 'do_no'thing' typo for key '{key}'")
            
            # Validate the action value before proceeding with key processing
            if current_action_value not in ("flap", "do_nothing"):
                # print(f"[Normalize WARN] Discarding rule with invalid action value '{value}' for key '{key}'")
                # log_failed_llm_attempt(str(llm_policy_dict_item), str(current_dict_to_process), 
                #                        f"Discarded rule with invalid action: {key}: {value}")
                continue # Skip this rule due to invalid action

            # Process the key itself
            parts = key.split('_')
            if not parts or not parts[0]: # Handles empty key string or key like "_"
                # print(f"[Normalize WARN] Discarding empty or malformed key string: '{key}'")
                continue

            # Check if all parts of the key have a valid prefix
            is_key_structurally_plausible = True 
            component_prefixes = {"pos:", "dist:", "velo:"}
            for part_str in parts:
                has_valid_prefix_for_this_part = False
                for prefix in component_prefixes:
                    if part_str.startswith(prefix):
                        has_valid_prefix_for_this_part = True
                        break
                if not has_valid_prefix_for_this_part:
                    is_key_structurally_plausible = False 
                    break 
            
            if not is_key_structurally_plausible:
                # print(f"[Normalize WARN] Discarding key '{key}' due to invalid part structure/prefix (e.g., '{part_str}' failed).")
                # log_failed_llm_attempt(str(llm_policy_dict_item), str(current_dict_to_process),
                #                        f"Discarded key with invalid part structure: {key}")
                continue # Skip this key

            # If key structure is plausible, sort components for multi-part keys
            final_key_for_policy = key # Default to original key (for single part keys)
            if len(parts) > 1:
                final_key_for_policy = "_".join(sorted(parts))
                # if key != final_key_for_policy:
                #      print(f"[Normalize DEBUG] Auto-sorted LLM key: '{key}' to '{final_key_for_policy}'")
            
            normalized_policy[final_key_for_policy] = current_action_value
            
        return normalized_policy
    # --- End of _normalize_and_correct_single_policy_dict helper ---


    # --- Main logic of normalize_policy based on expect_list ---
    # print(f"[Normalize DEBUG] Input to normalize_policy: {repr(policy_data_input)}, expect_list: {expect_list}")

    data_to_process_iteratively = None # This will be a list if expect_list is True
    single_dict_to_process = None    # This will be a dict if expect_list is False

    if expect_list:
        if isinstance(policy_data_input, dict):
            data_to_process_iteratively = [policy_data_input] # Wrap dict in list for processing
        elif isinstance(policy_data_input, list):
            data_to_process_iteratively = policy_data_input # Already a list
        else:
            # print(f"[Normalize ERROR] Initial pop: Expected list or dict, got {type(policy_data_input)}.")
            return None # Or an empty list: [] ? Let's return [] to match expected type.
    else: # Expecting a single dict (for mutation)
        if isinstance(policy_data_input, list):
            if len(policy_data_input) == 1 and isinstance(policy_data_input[0], dict):
                single_dict_to_process = policy_data_input[0] # Unwrap single dict from list
            else:
                # print(f"[Normalize ERROR] Mutation: Expected dict, got list not len 1 or not dict item.")
                return None 
        elif isinstance(policy_data_input, dict):
            single_dict_to_process = policy_data_input # Already a dict
        else:
            # print(f"[Normalize ERROR] Mutation: Expected dict, got {type(policy_data_input)}.")
            return None

    # Now, apply the processing
    if data_to_process_iteratively is not None: # Must be for initial population (expect_list was True)
        final_normalized_list = []
        for item_dict in data_to_process_iteratively:
            processed_item = _normalize_and_correct_single_policy_dict(item_dict)
            if processed_item: # Only add if successfully processed into a policy structure
                final_normalized_list.append(processed_item)
        # print(f"[Normalize DEBUG] Returning list from expect_list=True path: {final_normalized_list}")
        return final_normalized_list # Might be an empty list if all items failed processing
    
    elif single_dict_to_process is not None: # Must be for mutation
        result = _normalize_and_correct_single_policy_dict(single_dict_to_process)
        # print(f"[Normalize DEBUG] Returning single dict from expect_list=False path: {result}")
        return result # Might be None if the single dict failed processing
        
    # print(f"[Normalize DEBUG] Fallback: returning None from normalize_policy.")
    return None # Fallback if input was fundamentally incompatible or not processed


def validate_policy(policy):
    if not isinstance(policy, dict): 
        # print("[Validate DEBUG] Policy is not a dict")
        return False
    if "default" not in policy or policy["default"] not in ("flap", "do_nothing"):
        # print(f"[Validate DEBUG] Missing or invalid default: {policy.get('default')}")
        return False
    
    for key, value in policy.items():
        if value not in ("flap", "do_nothing"):
            # print(f"[Validate DEBUG] Invalid action '{value}' for key '{key}'")
            return False
        if key != "default":
            parts = key.split('_')
            
            # 1. Check for valid prefixes and values for each part
            # 2. Check for duplicate prefix types (e.g., two "pos:" parts)
            # 3. Check if multi-part keys are sorted (should be handled by normalize_policy for LLM output)
            
            seen_prefixes = set()
            for p_idx, p_val_str in enumerate(parts):
                valid_part_syntax = False
                current_part_prefix = None

                if p_val_str.startswith("pos:"):
                    val = p_val_str.split(":",1)[1]
                    if val in ["above", "aligned", "below"]: valid_part_syntax = True; current_part_prefix="pos:"
                elif p_val_str.startswith("dist:"):
                    val = p_val_str.split(":",1)[1]
                    if val in ["far", "medium", "close"]: valid_part_syntax = True; current_part_prefix="dist:"
                elif p_val_str.startswith("velo:"):
                    val = p_val_str.split(":",1)[1]
                    if val in ["rising", "stable", "falling"]: valid_part_syntax = True; current_part_prefix="velo:"
                
                if not valid_part_syntax:
                    # print(f"[Validate DEBUG] Invalid part syntax/value: '{p_val_str}' in key '{key}'")
                    return False 
                
                if current_part_prefix in seen_prefixes:
                    # print(f"[Validate DEBUG] Duplicate prefix type '{current_part_prefix}' in key '{key}'")
                    return False # Duplicate prefix type
                seen_prefixes.add(current_part_prefix)

            # Check if multi-part keys are now sorted (after normalization tried to fix them)
            if len(parts) > 1:
                if parts != sorted(parts):
                    # This means normalize_policy didn't sort it, or it wasn't from LLM (e.g. random mutation)
                    # AND the random mutation didn't sort it either.
                    # print(f"[Validate DEBUG] Key not alphabetically sorted: '{key}', should be '{'_'.join(sorted(parts))}'")
                    return False
    return True

# (get_action_from_policy, evaluate_fitness, crossover, generate_random_state_key, mutate,
#  _shared_game_eval_instance, run_flappy_game, play_game_with_policy, play_game_interactive
#  should be the same as your last verified complete version)

# Re-pasting key functions from your structure
def get_action_from_policy(policy, full_coded_state, action_map): # Assumes full_coded_state is sorted
    default_action_str = policy.get("default", "do_nothing"); default_numeric_action = action_map.get(default_action_str, 0)
    if full_coded_state in policy: return action_map.get(policy[full_coded_state], default_numeric_action)
    current_state_parts = full_coded_state.split('_') # Already sorted by get_coded_state
    if len(current_state_parts) == 3: # Check 2-part combinations
        for i in range(3):
            for j in range(i + 1, 3):
                # Parts are already sorted individually, and we pick them in order to form sorted 2-part keys
                two_part_key = "_".join(sorted([current_state_parts[i], current_state_parts[j]]))
                if two_part_key in policy: return action_map.get(policy[two_part_key], default_numeric_action)
    for one_part_key in current_state_parts: # Check 1-part combinations
        if one_part_key in policy: return action_map.get(policy[one_part_key], default_numeric_action)
    return default_numeric_action

def evaluate_fitness(policy, game_sim): # ... (same as before)
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

def generate_random_state_key(num_parts):
    pos_vals=["above","aligned","below"]; dist_vals=["far","medium","close"]; velo_vals=["rising","stable","falling"]
    components = []; available_types = [("pos", pos_vals), ("dist", dist_vals), ("velo", velo_vals)]
    if not 0 < num_parts <= len(available_types): num_parts = random.randint(1, len(available_types))
    selected_types = random.sample(available_types, num_parts)
    for prefix_str, val_list in selected_types:
        components.append(f"{prefix_str}:{random.choice(val_list)}")
    return "_".join(sorted(components))

def mutate(policy, use_llm=False):
    if use_llm and random.random() < 0.33:
        llm_mutated = llm_intelligent_mutation(policy.copy()) # BLOCKING
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

_shared_game_eval_instance = FlappyBirdGame()

def run_flappy_game(policy=None, interactive=False, max_frames=10000, render=True, return_fitness=False):
    game_instance = FlappyBirdGame()
    if render:
        if not pygame.get_init(): pygame.init()
        if not pygame.font.get_init(): pygame.font.init()
        WIDTH, HEIGHT = 400, FlappyBirdGame.GAME_SCREEN_HEIGHT
        screen = pygame.display.set_mode((WIDTH, HEIGHT)); pygame.display.set_caption("Flappy Evo")
        try: font = pygame.font.SysFont(None, 32)
        except pygame.error: font = pygame.font.Font(None, 32) 
    else: screen = None
    clock = pygame.time.Clock(); action_map = {"flap": 1, "do_nothing": 0}
    running = True; game_over = False
    while running and not game_over: # Main game loop
        action_to_take = 0 
        # Event handling
        if render: # Only process events if rendering to prevent issues in headless mode
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; game_over = True
                if interactive and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    action_to_take = 1
            if not running: break # Exit if QUIT event handled
        
        # Action determination
        if not interactive: # AI control
            if policy:
                state_code = game_instance.get_coded_state()
                action_to_take = get_action_from_policy(policy, state_code, action_map)
            # else action_to_take remains 0 (do_nothing if no policy)
        
        if not game_over: game_over = game_instance.step(action_to_take)

        if render and screen: # Rendering block
            screen.fill((135, 206, 235)); pipe_color = (34, 139, 34)
            uig_h = game_instance.pipe_gap_y - game_instance.gap_height // 2
            lig_y = game_instance.pipe_gap_y + game_instance.gap_height // 2
            pygame.draw.rect(screen, pipe_color, (game_instance.pipe_x, 0, game_instance.pipe_width, uig_h))
            pygame.draw.rect(screen, pipe_color, (game_instance.pipe_x, lig_y, game_instance.pipe_width, FlappyBirdGame.GAME_SCREEN_HEIGHT - lig_y))
            bird_color = (255, 215, 0); bird_rx = 60
            bird_ryc = int((game_instance.bird_y / FlappyBirdGame.BIRD_Y_MAX) * FlappyBirdGame.GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE)
            pygame.draw.circle(screen, bird_color, (bird_rx, bird_ryc), FlappyBirdGame.GAME_BIRD_PIXEL_RADIUS)
            if policy or interactive : # Show score/frames
                score_surf = font.render(f"Score: {game_instance.score}", True, (0,0,0)); screen.blit(score_surf, (10, 10))
                frames_surf = font.render(f"Frames: {game_instance.frames_survived}", True, (0,0,0)); screen.blit(frames_surf, (10, 40))
            pygame.display.flip(); clock.tick(60)

        if game_instance.frames_survived >= max_frames: game_over = True
        
        if game_over and running and render and interactive: # Game over message for human player
            font_big = pygame.font.SysFont(None, 48); msg_text = f"Game Over! Score: {game_instance.score}"
            msg = font_big.render(msg_text, True, (255,0,0)); msg_rect = msg.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(msg, msg_rect)
            prompt_text = "Press any key to return to menu..."
            prompt_surf = font.render(prompt_text, True, (200,200,200)); prompt_rect = prompt_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
            screen.blit(prompt_surf, prompt_rect); pygame.display.flip()
            waiting_for_key = True
            while waiting_for_key and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: waiting_for_key=False; running=False
                    if event.type == pygame.KEYDOWN: waiting_for_key=False
                clock.tick(15)
            running = False # Ensure main loop exits after game over message
        elif game_over and running and render: # Brief pause for AI crash
             pygame.time.wait(100) # Shorter pause
             running = False
        elif game_over: # Headless mode or non-interactive render
            running = False

    if return_fitness: return (game_instance.frames_survived * 10) + (game_instance.score * 500)
    return None

def play_game_with_policy(policy_to_play=None):
    loaded_policy = policy_to_play
    if loaded_policy is None:
        try:
            with open("best_policy.json", "r") as f: loaded_policy = json.load(f)
            print("[INFO] Loaded best_policy.json for playback.")
        except Exception as e: print(f"[ERROR] No best_policy.json: {e}. Train first."); return
    if not validate_policy(loaded_policy):
        print(f"[ERROR] Loaded policy invalid. Cannot play. Policy: {loaded_policy}"); return
    run_flappy_game(policy=loaded_policy, interactive=False, render=True, max_frames=20000)

def play_game_interactive():
    run_flappy_game(policy=None, interactive=True, render=True, max_frames=60000)

# --- train_policy (incorporating changes from previous successful merge) ---
# --- Thread state variables specific to initial population generation ---
import os
import time
import json
import shutil # For directory management
import threading
import csv
import pygame # Assuming pygame is used for the plot

# --- Directory for Per-Generation Best Policies (Ensure this is defined globally or accessible) ---
PER_GENERATION_BEST_DIR = "generation_best_policies" 

# --- Thread state variables specific to initial population generation (global for simplicity) ---
initial_pop_llm_thread = None
thread_result_container_init_pop = {'data': None} # Must be a dictionary for thread target to modify
thread_error_container_init_pop = {'error': None} # Must be a dictionary
thread_raw_response_container_init_pop = {'text': None} # Must be a dictionary

def train_policy():
    NUM_GENERATIONS = 100
    POPULATION_SIZE = 35 
    NUM_PARENTS = 10      

    # --- CSV Log file setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_log_filename = f"training_log_{timestamp}.csv"
    csv_log_file_path = os.path.join(os.getcwd(), csv_log_filename)
    try:
        with open(csv_log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv.writer(csvfile).writerow(['Generation', 'Policy_Index', 'Fitness', 'Policy_JSON'])
        print(f"[INFO] Training CSV log: {csv_log_file_path}")
    except IOError as e:
        print(f"[ERROR] CSV Log file creation failed: {e}. CSV Logging disabled.")
        csv_log_file_path = None

    # --- Per-Generation Best Policies Directory Setup ---
    run_timestamp_dir_name = time.strftime("%Y%m%d-%H%M%S_run")
    current_run_gen_best_dir = os.path.join(PER_GENERATION_BEST_DIR, run_timestamp_dir_name)
    try:
        os.makedirs(current_run_gen_best_dir, exist_ok=True)
        print(f"[INFO] Per-generation best policies for this run will be saved in: {current_run_gen_best_dir}")
    except OSError as e:
        print(f"[ERROR] Could not create dir for per-gen policies: {current_run_gen_best_dir}. Error: {e}")
        current_run_gen_best_dir = None 

    # --- Pygame plot setup ---
    if not pygame.get_init(): pygame.init()
    if not pygame.font.get_init(): pygame.font.init()
    PLOT_WIDTH, PLOT_HEIGHT = 800, 600
    plot_screen = pygame.display.set_mode((PLOT_WIDTH, PLOT_HEIGHT))
    pygame.display.set_caption("Flappy Evo - Training Progress")
    try:
        font_plot = pygame.font.SysFont(None, 24)
        font_plot_small = pygame.font.SysFont(None, 18)
    except pygame.error: 
        font_plot = pygame.font.Font(None, 24)
        font_plot_small = pygame.font.Font(None, 18)
    plot_margin = 60; graph_width = PLOT_WIDTH - 2 * plot_margin
    graph_height = PLOT_HEIGHT // 2 - plot_margin - 40
    
    # --- Training state variables ---
    fitness_history = []
    overall_best_policy = None
    overall_best_fitness = -float('inf')
    failed_policies_for_llm = [] 
    population = []
    
    print(f"\n--- Training Started: {NUM_GENERATIONS} generations, {POPULATION_SIZE} population ---")

    # --- Synchronous Initial Population Generation ---
    print("[INFO] Requesting initial population from LLM (synchronous call)...")
    # Display a "loading" message that will freeze until LLM responds
    plot_screen.fill((30, 30, 30))
    loading_text_surf = font_plot.render("LLM: Generating initial population (this will freeze GUI)...", True, (255,255,255))
    text_rect = loading_text_surf.get_rect(center=(PLOT_WIDTH // 2, PLOT_HEIGHT // 2))
    plot_screen.blit(loading_text_surf, text_rect)
    pygame.display.flip() # Show the loading message
    pygame.event.pump() # Process events once to ensure window is drawn before freeze

    num_to_request_from_llm = max(1, POPULATION_SIZE // 2) # Ask for half, at least 1
    prompt_for_initial_pop = llm_generate_initial_population_prompt_str(num_to_request_from_llm, failed_policies_for_llm)
    
    # Make the blocking call
    llm_data, raw_llm_response_text = call_ollama_blocking(prompt_for_initial_pop, LLM_TEMPERATURE_INIT_POP)

    if llm_data is None: # call_ollama_blocking handles logging the specific error
        print(f"[LLM Error/Warning - Initial Pop]: LLM call failed or returned no usable data. Raw: {raw_llm_response_text[:200] if raw_llm_response_text else 'N/A'}")
    
    normalized_policies = normalize_policy(llm_data, expect_list=True) 

    if normalized_policies and isinstance(normalized_policies, list):
        population = [p for p in normalized_policies if p and validate_policy(p)]
        print(f"  LLM provided {len(population)} valid policies after normalization and validation.")
        if len(population) == 0 and llm_data is not None:
            print(f"  LLM data (type: {type(llm_data)}) resulted in zero valid policies.")
            if raw_llm_response_text:
                log_failed_llm_attempt(prompt_for_initial_pop, raw_llm_response_text,
                                       f"Initial Pop - All items failed normalize/validate. Normalized output: {normalized_policies}")
    else:
        print(f"  Normalization of initial pop data failed or returned unexpected type. Norm_Output: {normalized_policies}")
        if llm_data is not None:
            log_failed_llm_attempt(prompt_for_initial_pop, 
                                   raw_llm_response_text or str(llm_data),
                                   f"Initial Pop - normalize_policy failed. Input type: {type(llm_data)}")
    
    if len(population) < POPULATION_SIZE:
        print(f"  Filling remaining {POPULATION_SIZE - len(population)} initial policies randomly.")
        while len(population) < POPULATION_SIZE:
            new_random_policy = generate_random_policy()
            if validate_policy(new_random_policy):
                population.append(new_random_policy)
            else:
                print("[WARN] Generated invalid random policy during fill. Using basic default.")
                population.append({"default": random.choice(["flap", "do_nothing"])}) # Fallback if gen random fails
    
    if not population:
        print("[CRITICAL ERROR] Population is empty after initial setup attempts. Halting training.")
        # pygame.display.quit() # Optional: close plot window before returning from error
        return None # Cannot proceed without a population

    print(f"[INFO] Initial population ready with {len(population)} policies. Starting generations...")
    
    # --- Main Evolutionary Loop ---
    quit_training_flag = False
    for gen_counter in range(NUM_GENERATIONS): # Use gen_counter to reflect actual EA cycles
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_training_flag = True; break
        if quit_training_flag: break

        plot_screen.fill((30, 30, 30)) # Clear screen for drawing this generation

        # --- Evolutionary Algorithm Steps ---
        # print(f"DEBUG: Starting EA steps for Gen {gen_counter + 1}") # Debug print

        fitness_scores = [evaluate_fitness(p, _shared_game_eval_instance) for p in population]
        
        if csv_log_file_path:
            try:
                with open(csv_log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    log_writer = csv.writer(csvfile)
                    for i, (p_item, f_s) in enumerate(zip(population, fitness_scores)):
                        log_writer.writerow([gen_counter + 1, i, f_s, json.dumps(p_item)]) # Use gen_counter
            except IOError as e: 
                csv_log_file_path = None 
                print(f"CSV Log write error: {e}. Further CSV logging disabled.")

        pop_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        
        if not pop_with_fitness:
            print(f"[ERROR] Generation {gen_counter+1}: No policies after fitness eval. Re-seeding randomly.")
            population = [generate_random_policy() for _ in range(POPULATION_SIZE)]
            if not population: print("[CRITICAL ERROR] Failed to re-seed random population. Stopping."); break
            pygame.time.wait(50) # Small pause before continuing
            continue 

        current_gen_best_p, current_gen_best_f = pop_with_fitness[0]

        if current_run_gen_best_dir and current_gen_best_p and validate_policy(current_gen_best_p):
            try:
                gen_best_filename = os.path.join(
                    current_run_gen_best_dir, 
                    f"gen_{gen_counter+1:04d}_fitness_{current_gen_best_f:.0f}_policy.json"
                )
                with open(gen_best_filename, "w", encoding="utf-8") as f:
                    json.dump(current_gen_best_p, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Could not save per-gen best policy for gen {gen_counter+1}: {e}")

        if current_gen_best_f > overall_best_fitness:
            overall_best_fitness = current_gen_best_f
            overall_best_policy = current_gen_best_p.copy()
            print(f"Gen {gen_counter+1}: New Overall Best Fitness: {overall_best_fitness:.0f}")
        
        fitness_history.append(current_gen_best_f)
        
        for p_item, score_val in pop_with_fitness:
            if score_val < 1000 and p_item not in failed_policies_for_llm: 
                 failed_policies_for_llm.append(p_item)
        if len(failed_policies_for_llm) > 5: failed_policies_for_llm = failed_policies_for_llm[-5:]

        parents = [p_data[0] for p_data in pop_with_fitness[:NUM_PARENTS]]
        next_generation = parents[:]
        
        while len(next_generation) < POPULATION_SIZE:
            p1 = random.choice(parents) if parents else generate_random_policy()
            p2 = random.choice(parents) if len(parents) > 1 else generate_random_policy()
            child = crossover(p1, p2)
            child = mutate(child, use_llm=True) # LLM mutation is still BLOCKING here
            if validate_policy(child): 
                next_generation.append(child)
            else: 
                next_generation.append(generate_random_policy())
        population = next_generation[:POPULATION_SIZE]

        # --- Plotting ---
        # (Plotting code: axes, Y-axis labels, fitness curve, text labels, best policy display)
        # Ensure all uses of 'gen' for display are now 'gen_counter + 1'
        pygame.draw.line(plot_screen,(200,200,200),(plot_margin,plot_margin),(plot_margin,plot_margin+graph_height),2)
        pygame.draw.line(plot_screen,(200,200,200),(plot_margin,plot_margin+graph_height),(plot_margin+graph_width,plot_margin+graph_height),2)
        if fitness_history:
            max_f_label=max(fitness_history,default=1.0); min_f_label=min(fitness_history,default=0.0)
            max_surf=font_plot_small.render(f"{max_f_label:.0f}",True,(200,200,200)); plot_screen.blit(max_surf,(plot_margin-40,plot_margin-7))
            min_surf=font_plot_small.render(f"{min_f_label:.0f}",True,(200,200,200)); plot_screen.blit(min_surf,(plot_margin-40,plot_margin+graph_height-7))
        if len(fitness_history)>1:
            max_f_hist=max(fitness_history,default=1.0); min_f_hist=min(fitness_history,default=0.0)
            f_range=max_f_hist-min_f_hist; f_range=f_range if f_range>0 else 1.0; points=[]
            num_gen_plotted=len(fitness_history)
            for i_fh,f_val in enumerate(fitness_history):
                x_coord=plot_margin+int((i_fh/max(1,num_gen_plotted-1))*graph_width) # Scale based on plotted points
                y_coord=plot_margin+graph_height-int(((f_val-min_f_hist)/f_range)*graph_height)
                points.append((x_coord,y_coord))
            if len(points)>1: pygame.draw.lines(plot_screen,(255,215,0),False,points,2)
        label_txt=f"Gen: {gen_counter+1}/{NUM_GENERATIONS} | Best: {overall_best_fitness:.0f} | Cur Best: {current_gen_best_f:.0f}" # Use gen_counter
        label_s=font_plot.render(label_txt,True,(255,255,255)); plot_screen.blit(label_s,(plot_margin,plot_margin+graph_height+10))
        if overall_best_policy:
            policy_disp_y=plot_margin+graph_height+40
            try:
                policy_lines=json.dumps(overall_best_policy,indent=1).splitlines()
                for i,line in enumerate(policy_lines):
                    if policy_disp_y+i*18 < PLOT_HEIGHT-10:
                        line_s=font_plot_small.render(line,True,(200,255,200)); plot_screen.blit(line_s,(plot_margin,policy_disp_y+i*18))
                    else: break
            except Exception: pass
        
        pygame.display.flip() 
        pygame.time.wait(10) # Can be adjusted or removed

    # --- End of Training Loop ---
    if quit_training_flag: 
        print("Training loop exited by user or critical error.")
    else: 
        print("\n--- Evolution Complete ---")

    if overall_best_policy:
        print("Final Best Policy:"); print(json.dumps(overall_best_policy,indent=2))
        try:
            with open("best_policy.json","w",encoding="utf-8") as f: json.dump(overall_best_policy,f,indent=2)
            print("Saved overall best policy to best_policy.json")
        except Exception as e: print(f"Error saving overall best_policy.json: {e}")
    else: 
        print("No valid overall best policy was found during training.")
    
    # If train_policy is called from main_menu, main_menu handles the final pygame.quit().
    # If this plot_screen was created by train_policy and might be the only display, quitting it here is an option.
    # However, it's generally better for the top-level app manager to control global Pygame state.
    # For now, we'll assume main_menu handles the ultimate pygame.quit().
    # If plot_screen is active:
    # if pygame.display.get_init() and pygame.display.get_surface() == plot_screen:
    #     pygame.display.quit() # Might be too aggressive if main_menu wants to show something else
    
    return overall_best_policy
# Ensure all other functions (FlappyBirdGame, LLM calls, helpers, main_menu, etc.)
# are correctly defined in the full script.

# --- main_menu function (ensure this is your complete version) ---
def main_menu():
    pygame.init()
    WIDTH, HEIGHT = 400, FlappyBirdGame.GAME_SCREEN_HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Evo - Main Menu")
    font_large = pygame.font.SysFont(None, 48); font_small = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock(); current_best_policy = None
    policy_file_checked = False; policy_file_valid = False

    os.makedirs("prompts", exist_ok=True)
    prompt_files_to_check = {
        POLICY_FORMAT_GUIDANCE_PATH: "Define policy format guidance.", EXAMPLE_POLICY_PATH: '{ "default": "flap" }',
        INITIAL_POP_TEMPLATE_PATH: "Define initial pop template. Use {n}, {guidance}, {failures}, {example}.",
        MUTATION_TEMPLATE_PATH: "Define mutation template. Use {policy}, {guidance}, {example}."}
    for p_file, p_content in prompt_files_to_check.items():
        if not os.path.exists(p_file):
            try:
                with open(p_file, "w", encoding="utf-8") as f: f.write(p_content)
                print(f"[INFO] Created placeholder prompt file: {p_file}")
            except Exception as e: print(f"[ERROR] Could not create placeholder {p_file}: {e}")

    def get_menu_options_list(): # Renamed
        nonlocal current_best_policy, policy_file_checked, policy_file_valid
        options = []
        if current_best_policy is None and not policy_file_checked:
            try:
                with open("best_policy.json", "r", encoding="utf-8") as f:
                    loaded_p = json.load(f)
                    if validate_policy(loaded_p): # Critical validation
                        current_best_policy = loaded_p; policy_file_valid = True
                        print("[INFO] Menu: Loaded valid best_policy.json.")
                    else: print("[INFO] Menu: best_policy.json found but invalid."); policy_file_valid = False
            except FileNotFoundError: print("[INFO] Menu: best_policy.json not found."); policy_file_valid = False
            except Exception as e: print(f"[INFO] Menu: Error loading best_policy.json: {e}"); policy_file_valid = False
            policy_file_checked = True
        
        if current_best_policy and validate_policy(current_best_policy):
            options.append(("Play Best Policy", "play_best"))
        options.extend([("Train New Policy", "train"), ("Play Yourself", "play_human"), ("Quit", "quit")])
        return options

    selected_index = 0; menu_running = True
    while menu_running:
        if not pygame.display.get_init() or pygame.display.get_surface() is None:
            screen = pygame.display.set_mode((WIDTH, HEIGHT)); pygame.display.set_caption("Flappy Evo - Main Menu")
            font_large = pygame.font.SysFont(None, 48); font_small = pygame.font.SysFont(None, 36)
        menu_items = get_menu_options_list()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: menu_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: selected_index = (selected_index - 1 + len(menu_items)) % len(menu_items) if menu_items else 0
                elif event.key == pygame.K_DOWN: selected_index = (selected_index + 1) % len(menu_items) if menu_items else 0
                elif event.key == pygame.K_RETURN and menu_items:
                    action_key = menu_items[selected_index][1]
                    if action_key == "play_best":
                        if current_best_policy and validate_policy(current_best_policy): play_game_with_policy(current_best_policy)
                        else: print("[Menu] No valid best policy to play.")
                    elif action_key == "train":
                        newly_trained = train_policy()
                        if newly_trained and validate_policy(newly_trained): current_best_policy = newly_trained
                        else: current_best_policy = None
                        policy_file_checked = False 
                    elif action_key == "play_human": play_game_interactive()
                    elif action_key == "quit": menu_running = False
                    menu_items = get_menu_options_list(); selected_index = min(selected_index, len(menu_items) -1) if menu_items else 0
        screen.fill((30,30,30)); title_surf = font_large.render("Flappy Evo",True,(255,255,255))
        screen.blit(title_surf, title_surf.get_rect(center=(WIDTH//2, 80)))
        for i, (text, _) in enumerate(menu_items):
            color = (255,255,0) if i == selected_index else (200,200,200)
            item_surf = font_small.render(text,True,color); screen.blit(item_surf, item_surf.get_rect(center=(WIDTH//2, 200+i*60)))
        pygame.display.flip(); clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main_menu()
