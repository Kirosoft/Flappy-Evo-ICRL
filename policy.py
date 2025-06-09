import os
import time
import json
import random
import csv
import pygame
from utils import log_failed_llm_attempt
from flappy_sim import _shared_game_eval_instance

PER_GENERATION_BEST_DIR = "generation_best_policies"
LLM_TEMPERATURE_INIT_POP = 0.7
NUM_GENERATIONS = 100
POPULATION_SIZE = 35 
NUM_PARENTS = 10      

def train_policy():
    from evolution import evaluate_fitness, crossover, mutate  # Local import to avoid circular import
    from llm import call_ollama_blocking, llm_generate_initial_population_prompt_str

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
        except Exception as e: print(f"Error saving overall_best_policy.json: {e}")
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

def generate_random_policy(): 
    policy = {"default": random.choice(["flap", "do_nothing"])}
    for _ in range(random.randint(0, 2)):
        key = generate_random_state_key(random.randint(1,3))
        if key and key not in policy: policy[key] = random.choice(["flap", "do_nothing"])
    return policy

def generate_random_state_key(num_parts):
    pos_vals=["above","aligned","below"]; dist_vals=["far","medium","close"]; velo_vals=["rising","stable","falling"]
    components = []; available_types = [("pos", pos_vals), ("dist", dist_vals), ("velo", velo_vals)]
    if not 0 < num_parts <= len(available_types): num_parts = random.randint(1, len(available_types))
    selected_types = random.sample(available_types, num_parts)
    for prefix_str, val_list in selected_types:
        components.append(f"{prefix_str}:{random.choice(val_list)}")
    return "_".join(sorted(components))

