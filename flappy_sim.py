# --- Gameplay Settings & LLM Settings (as before) ---
import random


PIPE_START_X_SETTING = 500; PIPE_SPEED_SETTING = 3; PIPE_RESPAWN_X_SETTING = 300 # ... etc.
PIPE_GAP_MIN_SETTING = 130; PIPE_GAP_MAX_SETTING = 170; BIRD_Y_MIN_SETTING = 0
BIRD_Y_MAX_SETTING = 100; GRAVITY_SETTING = 0.10; FLAP_STRENGTH_SETTING = -1.50
PIPE_MARGIN_Y_SETTING = 50


class FlappyBirdGame: 
    PIPE_START_X = PIPE_START_X_SETTING; PIPE_SPEED = PIPE_SPEED_SETTING; PIPE_RESPAWN_X = PIPE_RESPAWN_X_SETTING
    PIPE_GAP_MIN = PIPE_GAP_MIN_SETTING; PIPE_GAP_MAX = PIPE_GAP_MAX_SETTING; BIRD_Y_MIN = BIRD_Y_MIN_SETTING
    BIRD_Y_MAX = BIRD_Y_MAX_SETTING; GRAVITY = GRAVITY_SETTING; FLAP_STRENGTH = FLAP_STRENGTH_SETTING
    PIPE_MARGIN_Y = PIPE_MARGIN_Y_SETTING; GAME_SCREEN_HEIGHT = 600; GAME_BIRD_PIXEL_RADIUS = 20
    GAME_BIRD_LOGICAL_Y_TO_PIXEL_Y_EFFECTIVE_RANGE = GAME_SCREEN_HEIGHT-(2*GAME_BIRD_PIXEL_RADIUS)
    
    def __init__(self): 
        self.bird_y=50; self.bird_velocity=0; self.gravity=self.GRAVITY; self.flap_strength=self.FLAP_STRENGTH
        self.pipe_x=self.PIPE_START_X; self.score=0; self.frames_survived=0; self.pipe_width=60
        self.gap_height=random.randint(self.PIPE_GAP_MIN,self.PIPE_GAP_MAX); self.pipe_gap_y=self._random_gap_y(self.gap_height)
    
    def _random_gap_y(self, gap_height_pixels): 
        min_center_y=self.PIPE_MARGIN_Y+gap_height_pixels//2; max_center_y=self.GAME_SCREEN_HEIGHT-self.PIPE_MARGIN_Y-gap_height_pixels//2
        return random.randint(min_center_y,max_center_y) if min_center_y < max_center_y else self.GAME_SCREEN_HEIGHT//2
    
    def reset(self): self.__init__()

    def step(self, action): 
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
    
    def get_coded_state(self): 
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
    

_shared_game_eval_instance = FlappyBirdGame()