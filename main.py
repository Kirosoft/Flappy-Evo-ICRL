import os
from dotenv import load_dotenv
import json
import pygame 
from flappy_sim import FlappyBirdGame
from llm import EXAMPLE_POLICY_PATH, INITIAL_POP_TEMPLATE_PATH, MUTATION_TEMPLATE_PATH, POLICY_FORMAT_GUIDANCE_PATH
from policy import train_policy, validate_policy, get_action_from_policy

# --- Directory for Per-Generation Best Policies (Ensure this is defined globally or accessible) ---
PER_GENERATION_BEST_DIR = "generation_best_policies" 

load_dotenv()

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

    def get_menu_options_list(): 
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
