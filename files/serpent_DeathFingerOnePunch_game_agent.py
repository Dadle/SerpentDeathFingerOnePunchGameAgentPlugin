import time
from datetime import datetime
import os
import gc
import psutil
import offshoot
import pyautogui
import math
import numpy as np
from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from .helpers.terminal_printer import TerminalPrinter
from .helpers.memory_manager import MemoryManager

from plugins.SerpentDeathFingerOnePunchGameAgentPlugin.files.ddqn import dqn_core as dqn
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

plugin_path = offshoot.config["file_paths"]["plugins"]

# Constants used for zoom level
ZOOM_MAIN = "main"
ZOOM_BRAWLER = "brawler"
ZOOM_KILL_MOVE = "kill_move"


class SerpentDeathFingerOnePunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        print("Game agent initiating")
        super().__init__(**kwargs)

        # The game agent python process. Use this process element to monitor resource utilization
        self.process = psutil.Process(os.getpid())

        self.env_name = 'DeathFingerOnePunch'
        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)
        self.downscale_img_size = (int(self.game.window_geometry['height']/10),
                                   int(self.game.window_geometry['width']/10),
                                   2)  # This means 1 greyscale image and 1 greyscale motion-trace image

        # TerminalPrinter prints a list of instructions to cmd window, but ensures there is minmal stuttering in output
        self.printer = TerminalPrinter()

        # MemoryManager contains instructions for reading relevant memory addresses
        # These memory values are used as the basis for the agents reward function
        self.memory_manager = MemoryManager()

        ###             ###
        ### dqn SETUP  ###
        ###             ###

        self.input_mapping = {
            "NOOP": [],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "RIGHT": [KeyboardKey.KEY_RIGHT],
            "DOUBLE_RIGHT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_RIGHT],
            "DOUBLE_LEFT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_LEFT],
            "RIGHT_LEFT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_LEFT],
            "LEFT_RIGHT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_RIGHT]
        }

        # Action meaning for One Finger Death Punch
        self.ACTION_MEANING_OFDP = {
            0: "NOOP",
            1: "RIGHT",
            2: "LEFT",
            3: "DOUBLE_RIGHT",
            4: "DOUBLE_LEFT",
            5: "RIGHT_LEFT",
            6: "LEFT_RIGHT"
        }

        self.play_history = []

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.new_episode = True
        self.kill_count = 0
        self.miss_count = 0
        self.health = 10
        self.zoom_level = ZOOM_MAIN
        self.epsilon = 1.0
        self.not_playing_context_counter = 0
        self.episode_time = 0

        print("Game agent finished initiating")

    def setup_play(self):
        self.reset_game_state()
        context_classifier_path = f"{plugin_path}/SerpentDeathFingerOnePunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=self.window_dim)
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        self.machine_learning_models["context_classifier"] = context_classifier

        self.setup_ddqn()

        #To see the output of convolutional layers, set this to True
        if False:
            idx = 50  # np.argmax(self.replay_memory.rewards)
            self.plot_state(idx=idx)
            self.plot_layer_output(model=self.model, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)
            self.plot_layer_output(model=self.model, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)
            self.plot_layer_output(model=self.model, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)

    def setup_ddqn(self):
        dqn.checkpoint_base_dir = 'checkpoints'  # "D:\checkpoints"  # 'checkpoints_dqn'
        dqn.update_paths(env_name=self.env_name)

        # Setup DQN network and load from checkpoint if available
        self.agent = dqn.Agent(action_list=self.ACTION_MEANING_OFDP,
                               state_shape=self.downscale_img_size,
                               env_name=self.env_name,
                               training=True,  # TODO ----------------> switch and remember to set training again
                               render=True,)

        # Direct reference to the ANN in our agent for convenience
        self.model = self.agent.model

        # Direct reference to replay_memory for convenience
        self.replay_memory = self.agent.replay_memory

        self.episode_start_time = time.time()

    def handle_play(self, game_frame):
        self.context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        self.not_playing_context_counter += 1

        #print(self.context)
        if (self.context is None or self.context == "ofdp_game") and self.health > 0:
            #self.game_state["alive"].appendleft(1)
            self.make_a_move(game_frame)
            self.not_playing_context_counter = 0
            return
        elif self.kill_count > 10000:
            # This happens if the memory address for kill count changes for this episode
            # and we should just ignore the episode
            self.print_error()

        else:
            # Adding this code to avoid runs being ended early due to
            # context classifier getting the wrong context while playing
            if self.not_playing_context_counter < 4:
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

        move_time_start = time.time()

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()
        self.update_zoom_level(game_frame)
        self.kill_count = self.update_kill_count()
        self.health = self.update_health_counter(game_frame)  # self.update_health()
        #self.update_miss_counter(game_frame)

        if self.new_episode:
            #self.kill_count = 0

            self.episode_start_time = time.time()

            # Create a new motion-tracer for processing images from the
            # game-environment. Initialize with the first image-frame.
            # This resets the motion-tracer so the trace starts again.
            # This could also be done if end_life==True.
            self.motion_tracer = dqn.MotionTracer(image=game_frame.frame, target_size=self.downscale_img_size)

            # Increase the counter for the number of episodes.
            # This counter is stored inside the TensorFlow graph
            # so it can be saved and restored with the checkpoint.
            self.model.increase_count_episodes()

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
        #print(f"Clicking button: {buttons}")

        #test_time_start = time.time()
        self.input_controller.handle_keys(key_collection=buttons)
        #for button in buttons:
        #    self.input_controller.click(button=button)

        # test_time = time.time() - test_time_start
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
                                              reward=self.calculate_reward(),
                                              kill_count=self.kill_count,
                                              miss_count=self.miss_count,
                                              health=self.health)

        move_time = time.time() - move_time_start
        self.print_statistics(move_time=move_time, q_values=q_values)

    def calculate_reward(self):
        # reward is the number of kills made - number of health points lost since last state
        reward = 0.
        if len(self.replay_memory.episode_memory.states) > 0:
            reward += self.kill_count - self.replay_memory.episode_memory.kill_count[-1]
            reward += self.health - self.replay_memory.episode_memory.health[-1]
            #reward -= self.miss_count - self.replay_memory.episode_memory.miss_count[-1]
        return reward

    def end_episode(self):
        # Calculated survival time for current episode in seconds
        self.episode_time = time.time() - self.episode_start_time

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        self.add_printer_head()

        # The memory address is not stable and some times contain another memory pointer
        # When kill count contains a pointer we should not use this episode
        if self.kill_count < 1000:
            # Add the episode to the rest of replay memory and calculate statistics.
            # states, q_values, actions, rewards, end_episode = self.replay_memory.episode_memory.episode_end()
            self.replay_memory.add_episode_too_memory(self.episode_time, self.printer)

        # TODO: use this to check if replay memory still looks correct
        # Print the last memory states for debugging
        """
        
        self.plot_images(self.replay_memory.states[self.replay_memory.num_used-9:],
                         self.replay_memory.kill_count[self.replay_memory.num_used-9:],
                         self.replay_memory.health[self.replay_memory.num_used-9:])
        """

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
        #self.printer.add(f"Number of end_episode flags in memory: "
        #                 f"{np.sum(self.replay_memory.end_episode)}")

        self.printer.flush()

        if self.replay_memory.episode_statistics.num_episodes_completed % 25 == 0 \
                and self.replay_memory.episode_statistics.num_episodes_completed > 0\
                and self.agent.training:
            print("Saving replay memory checkpoint, please give me a minute...")
            self.replay_memory.store_memory_checkpoint()
            print("Finished storing replay memory for agent", self.env_name)

            # Train dqn at end of every few episodes
            self.train_dqn()

        self.reset_game_state()

        # Perform garbage collection after each episode to avoid memory leakage over time
        collected_garbage = gc.collect()
        #print("Garbage collector collected:", collected_garbage)

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
        self.printer.add(f"Current episode: {self.replay_memory.episode_statistics.num_episodes_completed}")
        self.printer.add("")

        self.printer.add(f"Average Last 10   Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_10, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_10, 2))}")
        self.printer.add(f"Average Last 100  Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_100, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_100, 2))}")
        self.printer.add(f"Average Last 1000 Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_1000, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_1000, 2))}")
        self.printer.add("")

    def print_statistics(self, move_time, q_values):
        episode_run_time_seconds = time.time() - self.episode_start_time
        state_count = len(self.replay_memory.episode_memory.states)
        #effective_apm = 0 if state_count else round(
        #    state_count / episode_run_time_seconds, 2)
        effective_apm = state_count / episode_run_time_seconds
        self.add_printer_head()

        #self.printer.add(f"Reading context: {self.context}")

        self.printer.add(f"AGENT DEATH FINGER ONE PUNCH THINKS:")
        self.printer.add(f"Decisions made this episode: {len(self.replay_memory.episode_memory.states)} "
                         f"\nAction: {self.ACTION_MEANING_OFDP[self.replay_memory.episode_memory.actions[-1]]}"
                         f"\nQ_values: {np.round(q_values, 1)}" # {np.round(self.replay_memory.episode_memory.q_values[-1], 10)}"
                         f"\nQ_values: [NOOP, LEFT, RIGHT, 2xLEFT, 2xRIGHT, R+L, L+R]"
                         f"\nQ_value is how good I think each move is right now")

        #current_reward = self.calculate_reward()
        current_reward = self.replay_memory.episode_memory.rewards[-1]
        reward_feedback = ''
        if current_reward > 0:
            reward_feedback = 'Rewarded'
        elif current_reward < 0:
            reward_feedback = 'punished'

        self.printer.add("")
        agent_mode = "training" if self.agent.training is True else "testing"
        self.printer.add(f"Agent is running in {agent_mode} mode")
        self.printer.add(f"Agent will be: {reward_feedback}")
        self.printer.add(f"Kill count   : {self.kill_count}")
        #self.printer.add(f"Miss count   : {self.miss_count}")
        self.printer.add(f"Player health: {self.health}")
        self.printer.add(f"Reward       : {current_reward}")
        self.printer.add("")

        # TEMP PRINT FOR DEBUGGING
        self.printer.add(f"Computing move in: {round(move_time, 3)} seconds")
        self.printer.add(f"Effective decisions per second: {round(effective_apm, 2)}")
        # self.printer.add(f"Episode clock time: "
        #                 f"{self.episode_time // 3600} hours, "
        #                 f"{(self.episode_time // 60) % 60} minutes, "
        #                 f"{self.episode_time % 60} seconds")
        #self.printer.add(f"States processed this episode: {len(self.replay_memory.episode_memory.states)}")

        if self.replay_memory.num_used > 0:
            self.printer.add("")
            self.printer.add(f"CURRENT RUN - TIME ALIVE: "
                             f"{round(episode_run_time_seconds, 2)} seconds "
                             f"and {self.kill_count} kills")
            self.printer.add(f"LAST    RUN - TIME ALIVE: "
                             f"{round(self.replay_memory.episode_statistics.last_episode_time, 2)} seconds "
                             f"and {self.replay_memory.episode_statistics.last_episode_kill_count} kills")
            self.printer.add(f"RECORD  RUN - TIME ALIVE: "
                             f"{round(self.replay_memory.episode_statistics.record_episode_time, 2)} seconds "
                             f"and {self.replay_memory.episode_statistics.record_kill_count} kills "
                             f"(Run {self.replay_memory.episode_statistics.record_episode})")

        # Finally print all above to the screen as one message (Less flickering)
        self.printer.flush()

    def print_error(self):
        self.add_printer_head()
        self.printer.add("Memory address changed for this run")
        self.printer.add("Agent will not be playing until kill count is correct again")
        self.printer.add("This should be resolved within one or two runs")
        self.printer.flush()

    def train_dqn(self):
        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        # How much of the replay-memory should be used.
        use_fraction = self.agent.replay_fraction.get_value(iteration=count_states)

        # When the replay-memory is sufficiently full.
        if self.replay_memory.is_full() \
                or self.replay_memory.used_fraction() > use_fraction:

            # Update all Q-values in the replay-memory through a backwards-sweep.
            # TODO: Swapped this approach with one only updating q_values for each episode when it is over
            # The idea is that the reward for each episode will not change after initial evaluation.
            # The reward will not change once an episode is completed, and we do not calculate
            # new q_values after an episode is finished, so why would we run through previous epiisodes
            # and growing the action values?
            # Answer: Because the Q values are growing due to the rolling out of reards backwards in time
            #         A Reward added on to the old q_value will grow the resulting action value we use for training
            #         to be higher than any previous Q value. This means Q values grow over time and are inherently
            #         unstable. We need to update all q_values to ensure they are on the correct scale,
            #         while ensuring that they do not grow out of control by only using a small batch size for training
            # self.replay_memory.update_all_q_values(self.printer)

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

    def plot_layer_output(self, model, layer_name, state_index, inverse_cmap=False):
        """
        Plot the output of a convolutional layer.

        :param model: An instance of the NeuralNetwork-class.
        :param layer_name: Name of the convolutional layer.
        :param state_index: Index into the replay-memory for a state that
                            will be input to the Neural Network.
        :param inverse_cmap: Boolean whether to inverse the color-map.
        """

        # Get the given state-array from the replay-memory.
        state = self.replay_memory.states[state_index]

        # Get the output tensor for the given layer inside the TensorFlow graph.
        # This is not the value-contents but merely a reference to the tensor.
        layer_tensor = model.get_layer_tensor(layer_name=layer_name)

        # Get the actual value of the tensor by feeding the state-data
        # to the TensorFlow graph and calculating the value of the tensor.
        values = model.get_tensor_value(tensor=layer_tensor, state=state)

        # Number of image channels output by the convolutional layer.
        num_images = values.shape[3]

        # Number of grid-cells to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_images))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

        print("Dim. of each image:", values.shape)

        if inverse_cmap:
            cmap = 'gray_r'
        else:
            cmap = 'gray'

        # Plot the outputs of all the channels in the conv-layer.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid image-channels.
            if i < num_images:
                # Get the image for the i'th output channel.
                img = values[0, :, :, i]

                # Plot image.
                ax.imshow(img, interpolation='nearest', cmap=cmap)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def update_kill_count(self):
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
        if tmp_health > self.health:
            return tmp_health
        else:
            return self.health

    def reset_game_state(self):
        self.kill_count = 0
        self.miss_count = 0
        self.health = 10
        self.zoom_level = ZOOM_MAIN
        self.username_entered = False
        self.episode_start_time = time.time()
        self.episode_end_time = None
        """
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
        """


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
            self.end_episode()
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
            self.episode_end_time = time.time()
            self.end_episode()
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

    @staticmethod
    def plot_images(images, kill_count, health, smooth=True):
        """
        plot both image and motion trace for 9 state images in a 3x3 matrix

        :param images: the state images to plot as numpy arrays
        :param kill_count: The current kill count for the state
        :param health: The current player health for the state
        :param smooth: Defaults to True. If True uses smooting when scaling images
        :return:
        """

        print(images.shape)

        # Create figure with sub-plots.
        fig, axes = plt.subplots(9, 2)

        hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)

        # Interpolation type.
        if smooth:
            interpolation = 'spline16'  # 'lanczos' # Alternative
        else:
            interpolation = 'nearest'

        for i, ax in enumerate(axes):
            # There may be less than 9 images, ensure it doesn't crash
            if i < len(images):
                # Plot the image and motion-trace for state i.
                # Game frame is index 0 and motion-trace is index 1 in each state
                for k in range(2):
                    # Plot image.
                    #ax.imshow(images[i], interpolation=interpolation)

                    #ax = axes.flat[0]
                    ax[k].imshow(images[i][:, :, k], vmin=0, vmax=255,
                              interpolation=interpolation, cmap='gray')

                    # Show kill count and health below the image
                    xlabel = "Kill count: {0}\nHealth: {1}".format(kill_count[i], health[i])

                    # Show the classes as the label on the x-axis
                    ax[k].set_xlabel(xlabel)

            # Remove ticks from the plot
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell
        plt.show()
