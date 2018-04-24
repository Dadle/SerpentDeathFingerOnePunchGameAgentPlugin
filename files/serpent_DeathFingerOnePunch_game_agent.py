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
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
from .helpers.terminal_printer import TerminalPrinter

from plugins.SerpentDeathFingerOnePunchGameAgentPlugin.files.ddqn import dqn_core as dqn
import cv2
import matplotlib.pyplot as plt

plugin_path = offshoot.config["file_paths"]["plugins"]

class SerpentDeathFingerOnePunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        print("Game agent initiating")
        super().__init__(**kwargs)

        self.process = psutil.Process(os.getpid())

        self.env_name = 'DeathFingerOnePunch'
        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)

        self.printer = TerminalPrinter()

        ###             ###
        ### ddqn SETUP  ###
        ###             ###

        self.input_mapping = {
            "NOOP": [],
            "LEFT": [MouseButton.LEFT],
            "RIGHT": [MouseButton.RIGHT]
        }

        # Action meaning for One Finger Death Punch
        self.ACTION_MEANING_OFDP = {
            0: "NOOP",
            1: "RIGHT",
            2: "LEFT"
        }

        self.key_mapping = {
            MouseButton.LEFT.name: "LEFT",
            MouseButton.RIGHT.name: "RIGHT"
        }

        self.play_history = []

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.new_episode = True

        print("Game agent finished initiating")

    def setup_play(self):
        self.kill_count = 0
        self.epsilon = 1.0
        self.game_state = {
            "health": collections.deque(np.full((8,), 10), maxlen=8),
            "nb_ennemies_hit": 0,
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

        self.setup_ddqn()
        idx = 50  # np.argmax(self.replay_memory.rewards)

        for i in range(-5, 3):
            # self.plot_state(idx=idx+i)
            continue
        # cv2.imshow("GameState", self.replay_memory.states[-1])
        # cv2.waitKey(0)

    def setup_ddqn(self):
        dqn.checkpoint_base_dir = "D:\checkpoints"  # 'checkpoints_dqn'
        dqn.update_paths(env_name=self.env_name)

        # Setup DQN network and load from checkpoint if available
        self.agent = dqn.Agent(action_list=self.ACTION_MEANING_OFDP,
                               env_name=self.env_name,
                               training=True,
                               render=True,
                               use_logging=False)

        # Direct reference to the ANN in our agent for convenience
        self.model = self.agent.model

        # Direct reference to replay_memory for convenience
        self.replay_memory = self.agent.replay_memory

        self.episode_start_time = datetime.now()

    def handle_play(self, game_frame):
        self.context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        #print(self.context)
        if self.context is None or self.context == "ofdp_game":
            self.game_state["alive"].appendleft(1)
            self.make_a_move(game_frame)
            return

        self.do_splash_screen_action(self.context)
        self.do_main_menu_actions(self.context)
        self.do_mode_menu_action(self.context)
        self.do_survival_menu_action(self.context)
        self.do_survival_pre_game_action(self.context)
        self.do_game_paused_action(self.context)
        self.do_game_end_highscore_action(self.context)
        self.do_game_end_score_action(self.context)

    def make_a_move(self, game_frame):
        """
        Use the Neural Network to decide which actions to take in each step through Q-value estimates.
        """

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        if self.new_episode:
            #self.kill_count = 0

            self.episode_start_time = datetime.now()

            # Create a new motion-tracer for processing images from the
            # game-environment. Initialize with the first image-frame.
            # This resets the motion-tracer so the trace starts again.
            # This could also be done if end_life==True.
            self.motion_tracer = dqn.MotionTracer(game_frame.frame)

            # Increase the counter for the number of episodes.
            # This counter is stored inside the TensorFlow graph
            # so it can be saved and restored with the checkpoint.
            self.model.increase_count_episodes()

            # Get the number of lives that the agent has left in this episode. TODO Do I need this?
            # num_lives = self.get_lives()

            self.new_episode = False

        # Process the image from the game-environment in the motion-tracer.
        # This will first be used in the next iteration of the loop.
        self.motion_tracer.process(image=game_frame.frame)

        # Get the state of the game-environment from the motion-tracer.
        # The state has two images: (1) The last image-frame from the game
        # and (2) a motion-trace that shows movement trajectories.
        state = self.motion_tracer.get_state()

        # Use the Neural Network to estimate the Q-values for the state.
        # Note that the function assumes an array of states and returns
        # a 2-dim array of Q-values, but we just have a single state here.
        q_values = self.model.get_q_values(states=[state])[0]

        # Determine the action that the agent must take in the game-environment.
        # The epsilon is just used for printing further below.
        action, self.epsilon = self.agent.epsilon_greedy.get_action(q_values=q_values,
                                                         iteration=count_states,
                                                         training=self.agent.training)

        # For SerpentAI we need to select the corresponding key input mapping taken by the input_controller object
        # action is just the index of the move, first get the text meaning of the move, then the key object
        action_meaning = self.ACTION_MEANING_OFDP[action]
        #print("MOVE:", action_meaning)
        buttons = self.input_mapping[action_meaning]
        #print("Clicking button:", buttons)
        for button in buttons:
            self.input_controller.click(button=button)
        #img, reward, end_episode, info = self.env.step(action=action)

        # Determine if a life was lost in this step. TODO Do I want this?
        # num_lives_new = self.get_lives()
        # end_life = (num_lives_new < num_lives)
        # num_lives = num_lives_new

        # Increase the counter for the number of states that have been processed.
        self.model.increase_count_states()

        # Add the state of the game-environment to the replay-memory.
        self.replay_memory.episode_memory.add(state=state,
                                              q_values=q_values,
                                              action=action,
                                              end_episode=False)

        self.print_statistics()

    def end_episde(self):
        # Add the episode's reward to a list and calculate statistics.
        # states, q_values, actions, rewards, end_episode = self.replay_memory.episode_memory.episode_end()
        self.replay_memory.add_episode_too_memory(self.model.get_count_episodes())
        self.agent.episode_rewards.append(self.replay_memory.rewards[-1])

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.get_count_episodes()

        # Mean reward of the last 30 episodes.
        if len(self.agent.episode_rewards) == 0:
            # The list of rewards is empty.
            reward_mean = 0.0
        else:
            reward_mean = np.mean(self.agent.episode_rewards[-30:])

        if self.agent.training:
            # Log reward to file.
            if self.agent.use_logging:
                self.agent.log_reward.write(count_episodes=count_episodes,
                                      count_states=count_states,
                                      reward_episode=self.replay_memory.episode_statistics.last_episode_time,
                                      reward_mean=reward_mean)

            # Print reward to screen.
            msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
            print(msg.format(count_episodes, count_states, self.epsilon,
                             self.replay_memory.episode_statistics.last_episode_time, reward_mean))

        #run_time = datetime.now() - self.started_at
        #episode_states = len(self.replay_memory.episode_memory.states) / 2  # FPS, TODO want to set dynamically
        #episode_hours = episode_states // 3600
        #episode_minutes = (episode_states // 60) % 60
        #episode_seconds = episode_states % 60

        #self.printer.add("")
        #self.printer.add("Capgemini Intelligent Automation - Death Finger One Punch agent")
        #self.printer.add("Reinforcement Learning: Training a DQN Agent")
        #self.printer.add("")
        #self.printer.add("\033c" + f"SESSION RUN TIME: "
        #                           f"{run_time.days} days, "
        #                           f"{run_time.seconds // 3600} hours, "
        #                           f"{episode_minutes} minutes, "
        #                           f"{run_time.seconds % 60} seconds")
        #self.printer.add(f"Episode lasted: {episode_hours} hours, "
        #                 f"{episode_minutes} minutes, "
        #                 f"{episode_seconds} seconds")


        self.add_printer_head()

        #self.printer.add(f"")
        #self.printer.add(f"Max q_value: {self.replay_memory.q_values.max(axis=1)}")
        #self.printer.add(f"Max reward: {self.replay_memory.rewards.max()}")
        #self.printer.add(f"Min q_value: {self.replay_memory.q_values.min(axis=1)}")
        #self.printer.add(f"Min reward: {self.replay_memory.rewards.min()}")
        #self.printer.add(f"Updated episode reward: {self.agent.episode_rewards}")

        self.printer.add(f"Replay memory is {round(self.replay_memory.used_fraction() * 100, 2)}% full")
        self.printer.add(f"Model trains when replay memory is more than "
                         f"{int(self.agent.replay_fraction.get_value(iteration=count_states) * 100)}% full")
        self.printer.add("")

        self.printer.flush()

        if self.replay_memory.episode_statistics.num_episodes_completed % 10 == 0 and self.replay_memory.episode_statistics.num_episodes_completed > 0:
            print("Saving replay memory checkpoint, please give me a minute...")
            self.replay_memory.store_memory_checkpoint()
            print("Finished storing replay memory for agent", self.env_name)

        # Train dqn at end of each episode
        self.train_dqn()
        self.reset_game_state()

    def add_printer_head(self):
        run_time = datetime.now() - self.started_at
        # serpent.utilities.clear_terminal()

        #self.printer.add("")
        self.printer.add("Capgemini Intelligent Automation - Death Finger One Punch agent")
        self.printer.add("Reinforcement Learning: Training a DQN Agent")
        self.printer.add("")

        self.printer.add("\033c" + f"SESSION RUN TIME: "
                                   f"{run_time.days} days, "
                                   f"{run_time.seconds // 3600} hours, "
                                   f"{(run_time.seconds // 60) % 60} minutes, "
                                   f"{run_time.seconds % 60} seconds")
        self.printer.add("")
        self.printer.add(f"Completed episodes: {self.replay_memory.episode_statistics.num_episodes_completed}")
        self.printer.add("")

        self.printer.add(f"Average survive time Last 10   Runs "
                         f"{round(self.replay_memory.episode_statistics.average_reward_10, 0)}")
        self.printer.add(f"Average survive time Last 100  Runs "
                         f"{round(self.replay_memory.episode_statistics.average_reward_100, 0)}")
        self.printer.add(f"Average survive time Last 1000 Runs: "
                         f"{round(self.replay_memory.episode_statistics.average_reward_1000, 0)}")
        #self.printer.add(f"Last episode reward: {self.agent.replay_memory.rewards[-1]}")
        self.printer.add("")

    def print_statistics(self):
        episode_time = datetime.now() - self.episode_start_time
        effective_fps = 0 if episode_time.seconds == 0 else round(
            len(self.replay_memory.episode_memory.states) / episode_time.seconds, 2)
        self.add_printer_head()

        #self.printer.add(f"Reading context: {self.context}")

        self.printer.add(f"AGENT DEATH FINGER ONE PUNCH THINKS:")
        self.printer.add(f"Moves made this episode: {len(self.replay_memory.episode_memory.states)} "
                         f"\nAction: {self.ACTION_MEANING_OFDP[self.replay_memory.episode_memory.actions[-1]]}"
                         f"\nQ_values: {self.replay_memory.episode_memory.q_values[-1]} "
                         f"\nQ_values represent: [NO OPERATION (NOOP), LEFT, RIGHT]"
                         f"\nQ_value is how good I think each move is right now")

        self.printer.add("")


        # TEMP PRINT FOR DEBUGGING
        self.printer.add(f"Effective FPS: {effective_fps}")
        self.printer.add(f"Episode clock time: "
                         f"{episode_time.seconds // 3600} hours, "
                         f"{(episode_time.seconds // 60) % 60} minutes, "
                         f"{episode_time.seconds % 60} seconds")
        self.printer.add(f"States processed this episode: {len(self.replay_memory.episode_memory.states)}")

        if self.replay_memory.num_used > 0:
            self.printer.add("")
            self.printer.add(f"CURR EPISODE TIME ALIVE: {len(self.replay_memory.episode_memory.states)/2} seconds")
            self.printer.add(f"LAST EPISODE TIME ALIVE: {self.replay_memory.episode_statistics.last_episode_time} seconds")

        self.printer.add(f"RECORD TIME ALIVE: "
                         f"{self.replay_memory.episode_statistics.record_episode_length} seconds "
                         f"(Run {self.replay_memory.episode_statistics.record_episode})")
                         #f"(Run {self.game_state['record_time_alive'].get('run')} "
                         #f"{'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")

        # Finally prtin all above to the screen as one message (Less flickering)
        self.printer.flush()

    def train_dqn(self):
        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.get_count_episodes()

        # How much of the replay-memory should be used.
        use_fraction = 0.1  # self.agent.replay_fraction.get_value(iteration=count_states)

        # When the replay-memory is sufficiently full.
        if self.replay_memory.is_full() \
                or self.replay_memory.used_fraction() > use_fraction:

            # Update all Q-values in the replay-memory through a backwards-sweep.
            self.replay_memory.update_all_q_values()  # TODO Should I do this?

            # Log statistics for the Q-values to file.
            if self.agent.use_logging:
                self.agent.log_q_values.write(count_episodes=count_episodes,
                                              count_states=count_states,
                                              q_values=self.replay_memory.q_values)

            # Get the control parameters for optimization of the Neural Network.
            # These are changed linearly depending on the state-counter.
            learning_rate = self.agent.learning_rate_control.get_value(iteration=count_states)
            loss_limit = self.agent.loss_limit_control.get_value(iteration=count_states)
            max_epochs = self.agent.max_epochs_control.get_value(iteration=count_states)

            # Perform an optimization run on the Neural Network so as to
            # improve the estimates for the Q-values.
            # This will sample random batches from the replay-memory.
            self.model.optimize(learning_rate=learning_rate,
                                loss_limit=loss_limit,
                                max_epochs=max_epochs)

            # Save a checkpoint of the Neural Network so we can reload it.
            self.model.save_checkpoint(count_states)

            # Reset the replay-memory. This throws away all the data we have
            # just gathered, so we will have to fill the replay-memory again.
            #self.replay_memory.reset()

    def plot_state(self, idx):
        """Plot the state in the replay-memory with the given index."""

        # Get the state from the replay-memory.
        state = self.replay_memory.states[idx]

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(1, 2)

        # Plot the image from the game-environment.
        ax = axes.flat[0]
        ax.imshow(state[:, :, 0], vmin=0, vmax=255,
                  interpolation='lanczos', cmap='gray')

        # Plot the motion-trace.
        ax = axes.flat[1]
        ax.imshow(state[:, :, 1], vmin=0, vmax=255,
                  interpolation='lanczos', cmap='gray')

        # This is necessary if we show more than one plot in a single Notebook cell.
        plt.show()

    def reset_game_state(self):
        self.username_entered = False
        self.episode_start_time = datetime.now()
        self.episode_end_time = None
        self.game_state["health"] = collections.deque(np.full((8,), 10), maxlen=8)
        self.game_state["nb_ennemies_hit"] = 0
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
            self.end_episde()
            #self.train_player_model()
            #self.train_dqn()
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
            self.new_episode = True

    def do_game_end_score_action(self, context):
        if context == "ofdp_game_end_score":
            # TODO: check score + nb of enemies killed.
            self.episode_end_time = time.time()
            self.end_episde()
            #self.train_player_model()
            #self.train_dqn()
            #print("I'M... dead.")
            #print("Waiting for button...")
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            self.new_episode = True
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