import time

# Constants used for zoom level
ZOOM_MAIN = "main"
ZOOM_BRAWLER = "brawler"
ZOOM_KILL_MOVE = "kill_move"


class GameState:
    """
    Contains the current game state with replay memory, memory manager and terminal printer access

    GameState objects are meant to be passed between system components to provide references to
    current game state and vital components like the ones mentioned above
    """

    def __init__(self, training, printer, memory_manager):
        self.printer = printer
        self.memory_manager = memory_manager

        self.new_episode = True
        self.kill_count = 0
        self.miss_count = 0
        self.health = 10
        self.zoom_level = ZOOM_MAIN
        self.training = training
        #self.epsilon = 0.9 if training else 0.05
        self.not_playing_context_counter = 0
        self.episode_time = 0
        self.started_at = time.time()
        self.username_entered = False
        self.episode_start_time = time.time()
        self.episode_end_time = None
        self.not_playing_context_counter = 0
        self.q_values = []
        self.episode_reward_total = 0
        self.agent_mode = "initial value"

        self.episode_statistics = []
        self.episode_kill_count = []

    def reset_game_state(self):
        self.kill_count = 0
        self.miss_count = 0
        self.health = 10
        self.zoom_level = ZOOM_MAIN
        self.username_entered = False
        self.episode_start_time = time.time()
        self.episode_end_time = None
        self.not_playing_context_counter = 0
        self.q_values = []
        self.episode_reward_total = 0
        self.episode_kill_count = []

    def read_kill_count(self):
        """
        Get kill count from game memory and check that it is larger than current value,
        if yes, return kill count from memory, else return current kill count
        In some game states memory says kill count is 0 falsely
        :return: current kill count
        """
        tmp_kill_count = self.memory_manager.read_kill_count()
        if tmp_kill_count > self.kill_count:
            return tmp_kill_count
        else:
            return self.kill_count

    def update_health(self):
        """
        Get health from game memory adn check that it is less than current value,
        if yes, return kill count from memory, else return current health
        :return: current player health
        """
        tmp_health = self.memory_manager.read_health()
        if tmp_health < self.health - 1:
            return self.health - 1
        else:
            return tmp_health

    def update_health_counter(self, game_frame):
        zoom_level = self.zoom_level

        if zoom_level == ZOOM_MAIN:
            first_x = 553
            first_y = 554
            last_health_x = 569
            last_health_y = 570
        elif zoom_level == ZOOM_BRAWLER:
            first_x = 606
            first_y = 607
            last_health_x = 622
            last_health_y = 623
        elif zoom_level == ZOOM_KILL_MOVE:
            # Can't get any new modification on health here
            # return
            first_x = 553
            first_y = 554
            last_health_x = 569
            last_health_y = 570

        current_health = 0

        for nb_health in range(0, 9):
            region_health = game_frame.frame[first_x:first_y, 786 - (35 * nb_health):787 - (35 * nb_health)]
            if region_health[0, 0, 0] > 200:
                current_health += 1

        health_last = game_frame.frame[last_health_x:last_health_y, 475:476]
        # "REGION": (569, 475, 570, 476)

        if health_last[0, 0, 0] > 200:
            current_health += 1

        if current_health == 0 and self.health != 1:
            # At times the zoom level or other factors can affent how health is read
            # This code avoids that rounds are ended cue to this issue
            return self.health
        else:
            return current_health

    """
    def update_bonus_mode_and_hits(self, game_frame):
        for nb_hits in range(0, 4):
            region_hit = game_frame.frame[618:619, 714 - (50 * nb_hits):715 - (50 * nb_hits)]
            if sum(region_hit[0, 0]) == 306:
                self.game_state["bonus_hits"] += 1

        if self.game_state["bonus_hits"] > 0:
            self.game_state["bonus_mode"] = True
        self.game_state["bonus_mode"] = False
    """

    # Check the zoom on game screen. "main" is the normal game, "brawler" when
    # a brawler is fighting, "kill_move" when the character does a kill move
    def update_zoom_level(self, game_frame):
        check_zoom_mode = game_frame.frame[563:564, 639:640]
        sum_pixels = sum(check_zoom_mode[0, 0])
        if sum_pixels > 300:
            self.zoom_level = ZOOM_MAIN
        elif sum_pixels == 300:
            self.zoom_level = ZOOM_BRAWLER
        elif sum_pixels < 300:
            self.zoom_level = ZOOM_KILL_MOVE
