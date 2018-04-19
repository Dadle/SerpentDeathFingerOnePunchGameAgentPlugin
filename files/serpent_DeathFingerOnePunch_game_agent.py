from serpent.game_agent import GameAgent
import time
from datetime import datetime
import os
import psutil
import gc
import offshoot
import pyautogui
import collections
import numpy as np
import serpent.utilities
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier
from plugins.SerpentDeathFingerOnePunchGameAgentPlugin.files.keras_model import KerasDeepPlayer

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

plugin_path = offshoot.config["file_paths"]["plugins"]

# Constants used for zoom level
ZOOM_MAIN = "main"
ZOOM_BRAWLER = "brawler"
ZOOM_KILL_MOVE = "kill_move"

class SerpentDeathFingerOnePunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        print("Game agent initiating")
        super().__init__(**kwargs)

        self.process = psutil.Process(os.getpid())

        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)
        print("Game runs in native resolution:", self.window_dim)
        self.memory_timeframe = 6  #at 2 FPS means 3 seconds of history

        #self.player_model = KerasDeepPlayer(time_dim=(self.memory_timeframe,),
        #                                    game_frame_dim=self.window_dim)

        ###             ###
        ### DDQN SETUP  ###
        ###             ###

        self.input_mapping = {
            "LEFT": [MouseButton.LEFT],
            "RIGHT": [MouseButton.RIGHT]
        }

        self.key_mapping = {
            MouseButton.LEFT.name: "LEFT",
            MouseButton.RIGHT.name: "RIGHT"
        }
        movement_action_space = KeyboardMouseActionSpace(
            default_keys=[None, "LEFT", "RIGHT"]
        )

        movement_model_file_path = "datasets/ofdp_direction_dqn_0_1_.hp5"
        self.dqn_movement = DDQN(
            model_file_path=movement_model_file_path if os.path.isfile(movement_model_file_path) else None,
            input_shape=(72, 128, 3),
            input_mapping=self.input_mapping,
            action_space=movement_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            model_learning_rate=0.001,#1e-4,
            initial_epsilon=1,
            final_epsilon=0.0001,
            override_epsilon=False
        )

        self.play_history = []

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        print("Game agent initializing Redis waiting for frames to flow")
        self.game_frame_buffer = FrameGrabber.get_frames(
            [2, 1, 0, 3],
            # Last frame is the next state while final frame should be first to allow extension of history
            frame_type="PIPELINE"
        )
        print("Game agent redis OK")

        print("Game agent finished initiating")

    def setup_play(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 10), maxlen=8),
            "nb_ennemies_hit": 0,
            "zoom_level": ZOOM_MAIN,
            "bonus_mode": False,
            "bonus_hits": 4,
            "nb_miss": 0,
            "miss_failsafe": 2,
            # DQN variables
            "current_run": 1,
            "current_run_started_at": datetime.utcnow(),
            "current_run_duration": None,
            "current_run_steps": 0,
            "run_reward": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run": 0,
            "last_run_duration": 0,
            "last_run_duration_actual": None,
            "last_run_distance": 0.0,
            "last_run_coins_collected": 0,
            "record_duration": None,
            "record_duration_actual": 0,
            "record_run": 0,
            "record_distance": 0.0,
            "alive": collections.deque(np.full((8,), 4), maxlen=8),
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "random_distance_travelled": 0.0
        }

        self.reset_game_state()
        context_classifier_path = f"{plugin_path}/SerpentDeathFingerOnePunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=self.window_dim)
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):
        gc.disable()
        self.context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        #print(self.context)
        if self.context is None or self.context == "ofdp_game":
            #print("FIGHT!")
            self.game_state["alive"].appendleft(1)
            #print("There's nothing there... Waiting...")
            self.make_a_move()
            return

        self.do_splash_screen_action(self.context)
        self.do_main_menu_actions(self.context)
        self.do_mode_menu_action(self.context)
        self.do_survival_menu_action(self.context)
        self.do_survival_pre_game_action(self.context)
        self.do_game_paused_action(self.context)
        self.do_game_end_highscore_action(self.context)
        self.do_game_end_score_action(self.context)

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

        ###
        # Old manual code
        ###
        # self.game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3, 4, 5], frame_type="PIPELINE")
        # frame_buffer = self.game_frame_buffer.frames
        # move_per_timestep = self.player_model.decide(frame_buffer)
        # move = move_per_timestep[len(move_per_timestep) - 1]
        # self.play_history.append({"moves": move_per_timestep, "frames": frame_buffer, 'sequence_end_time': time.time()})

        # if move is not None:
        #    print("MOVE:", move)
        #    self.input_controller.click(button=move)
        # else:
        #    print("MOVE: Waiting")

    def make_a_move(self):

        ###
        # DDQN execution code
        ###
        self.game_frame_buffer = FrameGrabber.get_frames(
            [0, 1, 2, 3],
            frame_type="PIPELINE"
        )
        frame = self.game_frame_buffer.frames[0].frame
        if self.dqn_movement.frame_stack is None or self.dqn_movement.mode == "RUN":
            #print("FRAME BUFFER CONTAINS:")
            #print(frame.shape)
            self.dqn_movement.build_frame_stack(frame)
            #self.dqn_movement.build_frame_stack(game_frame.ssim_frame)
        else:
            #print("FRAME BUFFER CONTAINS:")
            #print(self.game_frame_buffer.frames[0].frame.shape)

            if self.dqn_movement.mode == "TRAIN":
                reward = self._calculate_reward()

                self.game_state["run_reward"] += reward

                self.dqn_movement.append_to_replay_memory(
                    self.game_frame_buffer,
                    reward,
                    terminal=self.game_state["alive"] == 0
                )
                # Every 2000 steps, save latest weights to disk
                if self.dqn_movement.current_step % 2000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/dqn_movement"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_movement.current_step % 20000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/dqn_movement",
                        is_checkpoint=True
                    )

            run_time = datetime.now() - self.started_at

            serpent.utilities.clear_terminal()
            print("\033c" + f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
            print("")

            print("Reading context:", self.context)

            print("MOVEMENT NEURAL NETWORK:\n")
            self.dqn_movement.output_step_data()

            print("")
            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward'], 2)}")
            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT PLAYER ALIVE: {self.game_state['alive'][0]}")

            print("")
            # print(f"AVERAGE ACTIONS PER SECOND: {round(self.game_state['average_aps'], 2)}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")
            # print(f"LAST RUN COINS: {self.game_state['last_run_coins'][0]})

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")
            print("")
            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")

        self.dqn_movement.pick_action()
        self.dqn_movement.generate_action()

        keys = self.dqn_movement.get_input_values()
        print("")
        print("Chosen move:", " + ".join(list(map(lambda k: self.key_mapping.get(k.name), keys))))

        print ("")
        total, available, percent, used, free = psutil.virtual_memory()
        proc = self.process.memory_info()[1]
        print('process = %s | total = %s | available = %s | used = %s | free = %s | percent free = %s'
              % (proc/1000, total/1000, available/1000, used/1000, free/1000, percent))

        print("")
        print("")
        print("", self.dqn_movement.replay_memory.tree.)

        for key in keys:
            self.input_controller.click(button=key)

        if self.dqn_movement.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_movement.erode_epsilon(factor=2)

        self.dqn_movement.next_step()

        self.game_state["current_run_steps"] += 1



    def train_player_model(self):
        episode = []
        for sequence in self.play_history:
            score_per_timestep = self.player_model.evaluate_move(sequence['moves'], sequence['frames'], sequence['sequence_end_time'], self.episode_start_time, self.episode_end_time)
            episode.append({'frames': sequence['frames'], 'scores': score_per_timestep, 'sequence_end_time': sequence['sequence_end_time']})
        self.player_model.train_episode(episode, self.episode_start_time, self.episode_end_time)

    def reset_game_state(self):
        self.username_entered = False
        self.episode_start_time = time.time()
        self.episode_end_time = None
        self.game_state["health"] = collections.deque(np.full((8,), 10), maxlen=8)
        self.game_state["nb_ennemies_hit"] = 0
        self.game_state["zoom_level"] = ZOOM_MAIN
        self.game_state["bonus_mode"] = False
        self.game_state["bonus_hits"] = 4
        self.game_state["nb_miss"] = 0
        self.game_state["miss_failsafe"] = 2
        #DQN variables
        #self.game_state["current_run"] = 1
        self.game_state["current_run_started_at"] = datetime.utcnow()
        self.game_state["current_run_duration"] = None
        self.game_state["current_run_steps"] = 0
        self.game_state["run_reward"] = 0
        self.game_state["run_future_rewards"] = 0
        self.game_state["run_predicted_actions"] = 0
        self.game_state["run_timestamp"] = datetime.utcnow()
        self.game_state["alive"] = collections.deque(np.full((8,), 4), maxlen=8)
        #self.game_state["random_time_alive"] = None
        self.game_state["random_distance_travelled"] = 0.0

    def train_dqn(self):
        serpent.utilities.clear_terminal()
        timestamp = datetime.utcnow()
        #print("Timestamp when training starts:", timestamp)
        #print("Timestamp when round started:", self.game_state["run_timestamp"])

        gc.enable()
        gc.collect()
        gc.disable()

        # Set display stuff TODO
        timestamp_delta = timestamp - self.game_state["run_timestamp"]
        #print("Delta timestamp:", timestamp_delta.seconds)
        self.game_state["last_run_duration"] = int(timestamp_delta.seconds)

        if self.dqn_movement.mode in ["TRAIN", "RUN"]:
            # Check for Records
            if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                self.game_state["record_time_alive"] = {
                    "value": self.game_state["last_run_duration"],
                    "run": self.game_state["current_run"],
                    "predicted": self.dqn_movement.mode == "RUN"
                }

        else:
            self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
            self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

        self.game_state["current_run_steps"] = 0

        #self.input_controller.release_key(KeyboardKey.KEY_SPACE) #TODO verify or delete

        if self.dqn_movement.mode == "TRAIN":
            for i in range(8):
                serpent.utilities.clear_terminal()
                print(f"")
                print(f"")
                print(f"TRAINING ON MINI-BATCHES: {i + 1}/8")
                print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                self.dqn_movement.train_on_mini_batch()

        self.game_state["run_timestamp"] = datetime.utcnow()
        self.game_state["current_run"] += 1
        self.game_state["run_reward_movement"] = 0
        self.game_state["run_predicted_actions"] = 0
        self.game_state["alive"] = collections.deque(np.full((8,), 4), maxlen=8)

        if self.dqn_movement.mode in ["TRAIN", "RUN"]:
            if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                if self.dqn_movement.type == "DDQN":
                    self.dqn_movement.update_target_model()

            if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                self.dqn_movement.enter_run_mode()
            else:
                self.dqn_movement.enter_train_mode()

        #self.input_controller.tap_key(KeyboardKey.KEY_SPACE) #TODO verify or delete
        time.sleep(5)

        return None

    def _calculate_reward(self):
        reward = 0

        reward += (-0.5 if self.game_state["alive"][2] < self.game_state["alive"][3] else 0.05)
        # reward += (0.5 if (self.game_state["coins"][0] - self.game_state["coins"][1]) >= 1 else -0.05)

        if self.context == "ofdp_game_end_highscore":
            reward += reward*500

        return reward


    #Context actions to get into the game
    def do_splash_screen_action(self, context):
        if context == "ofdp_splash_screen":
            #print("Boring part, just click on \"Play\"")
            self.input_controller.move(x=650, y=460)
            self.input_controller.click()
            time.sleep(60)

    def do_main_menu_actions(self, context):
        if context == "ofdp_main_menu":
            #print("Boring part 2, just click on \"Play\"... again")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            time.sleep(3)

    def do_mode_menu_action(self, context, game_mode="MODE_MENU_SURVIVAL"):
        if context == "ofdp_mode_menu":
            #print("What to choose ? Oh ! Survival !")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region=game_mode
            )
            time.sleep(3)

    def do_survival_menu_action(self, context, game_mode="SURVIVAL_MENU_BUTTON_TOP"):
        if context == "ofdp_survival_menu":
            #print("SURVIVAL!")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region=game_mode
            )
            time.sleep(2)

    def do_survival_pre_game_action(self, context):
        if context == "ofdp_survival_pre_game":
            self.survival_start_time = time.time()
            #print("Who needs skill. Let's play !")

            self.reset_game_state()
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_PRE_GAME_START_BUTTON"
            )
            time.sleep(2)


    def do_game_paused_action(self, context):
        # TODO: add click for quitting or resume.
        if context == "ofdp_game_paused":
            print("PAUSING AND WAITING")
            time.sleep(1)

    def do_game_end_highscore_action(self, context):
        if context == "ofdp_game_end_highscore":
            self.survival_end_time = time.time()
            #self.train_player_model()
            self.train_dqn()
            print("NEW HIGHSCORE!")

            if not self.username_entered:
                for letter in ["HS_LETTER_D", "HS_LETTER_A", "HS_LETTER_D", "HS_LETTER_L", "HS_LETTER_E"]:
                    self.input_controller.click_screen_region(
                        button=MouseButton.LEFT,
                        screen_region=letter
                    )
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()

                self.username_entered = True

            print("Entering nickname, done !")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="HS_OK"
            )

    def do_game_end_score_action(self, context):
        if context == "ofdp_game_end_score":
            # TODO: check score + nb of enemies killed.
            self.episode_end_time = time.time()
            #self.train_player_model()
            self.train_dqn()
            print("I'M... dead.")
            #print("Waiting for button...")
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            time.sleep(3)


    #Util methods I might use
    def update_health_counter(self, game_frame):
        zoom_level = self.game_state["zoom_level"]

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

        if -1 <= self.game_state["health"][0] - current_health <= 1:
            self.game_state["health"].appendleft(current_health)

    def update_miss_counter(self, game_frame):
        miss_region = rgb2gray(game_frame.frame[357:411, 570:710])
        self.game_state["miss_failsafe"] -= 1
        # print(sum(sum(miss_region)))
        if 3400 < sum(sum(miss_region)) < 3500 and self.game_state["miss_failsafe"] < 0 and self.game_state["zoom_level"] is ZOOM_MAIN:
            self.game_state["nb_miss"] += 1
            self.game_state["miss_failsafe"] = 2

    def update_bonus_mode_and_hits(self, game_frame):
        for nb_hits in range(0, 4):
            region_hit = game_frame.frame[618:619, 714 - (50 * nb_hits):715 - (50 * nb_hits)]
            if sum(region_hit[0, 0]) == 306:
                self.game_state["bonus_hits"] += 1

        if self.game_state["bonus_hits"] > 0:
            self.game_state["bonus_mode"] = True
        self.game_state["bonus_mode"] = False

    # Check the zoom on game screen. "main" is the normal game, "brawler" when
    # a brawler is fighting, "kill_move" when the character does a kill move
    def update_zoom_level(self, game_frame):
        check_zoom_mode = game_frame.frame[563:564, 639:640]
        sum_pixels = sum(check_zoom_mode[0, 0])
        if sum_pixels > 300:
            self.game_state["zoom_level"] = ZOOM_MAIN
        elif sum_pixels == 300:
            self.game_state["zoom_level"] = ZOOM_BRAWLER
        elif sum_pixels < 300:
            self.game_state["zoom_level"] = ZOOM_KILL_MOVE