import time
import os
import gc
import psutil
import offshoot
import torch
import tensorflow as tf
from keras import backend as Keras_backend
from datetime import datetime
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from .helpers.memory_manager import MemoryManager
from .helpers.game_state import GameState
from .helpers.image_plotter import ImagePlotter

from .rainbow_dqn import rainbow_agent
from .rainbow_dqn.replay_memory import ReplayMemory
from .rainbow_dqn.rainbow_arguments import RainbowArguments
from .rainbow_dqn.test import test

import matplotlib.pyplot as plt

plugin_path = offshoot.config["file_paths"]["plugins"]


class SerpentDeathFingerOnePunchGameAgent(GameAgent):
    """
    GameAgent class contains all code relevant to the specific game being played.

    ANy code related to AI methods should be included in other files
    """

    def __init__(self, **kwargs):
        print("Game agent initiating")
        super().__init__(**kwargs)

        # The game agent python process. Use this process element to monitor resource utilization
        self.process = psutil.Process(os.getpid())

        self.env_name = 'DeathFingerOnePunch'
        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)

        # To avoid issues with GPU memory when using both Keras and Pytorch at the same time
        # Forcing Keras to run on CPU since this is a pretrained classifier that does not need GPU acceleration
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
                                #intra_op_parallelism_threads=4,
                                #inter_op_parallelism_threads=4, allow_soft_placement=True,
                                #device_count={'CPU': 4, 'GPU': 0})
        session = tf.Session(config=config)
        Keras_backend.set_session(session)

        # MemoryManager contains instructions for reading relevant memory addresses
        # These memory values are used as the basis for the agents reward function
        self.memory_manager = MemoryManager()

        self.image_plotter = ImagePlotter()

        self.input_mapping = {
            "NOOP": [],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "RIGHT": [KeyboardKey.KEY_RIGHT]

        }
        """
        "DOUBLE_RIGHT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_RIGHT],
        "DOUBLE_LEFT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_LEFT],
        "RIGHT_LEFT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_LEFT],
        "LEFT_RIGHT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_RIGHT]
        """

        # Action meaning for One Finger Death Punch
        self.ACTION_MEANING_OFDP = {
            0: "NOOP",
            1: "LEFT",
            2: "RIGHT"
        }
        """
        3: "DOUBLE_RIGHT",
        4: "DOUBLE_LEFT",
        5: "RIGHT_LEFT",
        6: "LEFT_RIGHT"
        """

        self.play_history = []

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        print("Game agent finished initiating")

    def setup_play(self):
        self.machine_learning_models["context_classifier"] = self.setup_context_classifier()

        self.username_entered = False
        self.training_loss = 0

        # Agent
        self.rainbow_arguments = RainbowArguments()
        # Setup DQN network and load from checkpoint if available
        self.replay_memory = ReplayMemory(args=self.rainbow_arguments,
                                          capacity=self.rainbow_arguments.memory_capacity,
                                          memory_type='training')
        self.agent = rainbow_agent.Agent(self.rainbow_arguments, len(self.input_mapping), self.replay_memory)
        self.agent.train()  # To run in evaluation mode, change this to self.agent.eval()

        self.priority_weight_increase = (1 - self.rainbow_arguments.priority_weight) / \
                                        (self.rainbow_arguments.T_max - self.rainbow_arguments.learn_start)

        # Construct validation memory
        self.val_mem = ReplayMemory(args=self.rainbow_arguments,
                                    capacity=self.rainbow_arguments.evaluation_size,
                                    memory_type='validation')

        # Convenient access to vital objects
        # Direct reference to the ANN in our agent for convenience
        #self.model = self.agent.model
        # Direct reference to replay_memory for convenience
        #self.replay_memory = self.agent.replay_memory

        self.game_state = GameState(training=self.agent.online_net.training,
                                    printer=self.agent.printer,
                                    memory_manager=self.memory_manager)
        #self.agent.game_state = self.game_state
        self.replay_memory.game_state = self.game_state
        self.val_mem.game_state = self.game_state
        self.agent.printer.game_state = self.game_state

        self.episode_end = False
        self.evaluation_run_count = 0
        self.last_updated_net_time = "Not updated yet"
        self.last_updated_net_episode = self.replay_memory.episode_count

        # render the output of the cnn layers for a state to see what features are learned
        #self.image_plotter.render_cnn_layer_outputs(self.game_state)


    def setup_context_classifier(self):
        context_classifier_path = f"{plugin_path}/SerpentDeathFingerOnePunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=self.window_dim)
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        return context_classifier

    def handle_play(self, game_frame):
        self.move_time_start = time.time()
        self.game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        frame_buffer = self.game_frame_buffer.frames

        context_frame = FrameGrabber.get_frames([0], frame_type="FULL").frames[0]
        context = self.machine_learning_models["context_classifier"].predict(context_frame.frame)
        self.game_state.current_context = context
        self.game_state.not_playing_context_counter += 1
        self.move_time = time.time() - self.move_time_start

        #print(context)
        if (context is None or context in ["ofdp_playing", "ofdp_game"]) and self.game_state.health > 0:
            # If we are currently playing an episode
            if self.game_state.read_kill_count() > 10000:
                # If kill_count gets really large during the episode we should ignore the episode
                # This happens if the memory address for kill count changes for this episode
                # and we should just ignore the episode
                self.agent.printer.print_error()
                return

            self.make_a_move(frame_buffer, context_frame)
            self.game_state.not_playing_context_counter = 0
            return
        else:
            # This is a hack to avoid runs being ended early due to
            # context classifier getting the wrong context while playing
            if self.game_state.not_playing_context_counter < 5:
                return

        # Navigate context menu if identified
        self.do_splash_screen_action(context)
        self.do_main_menu_actions(context)
        self.do_mode_menu_action(context)
        self.do_survival_menu_action(context)
        self.do_survival_pre_game_action(context)
        self.do_game_paused_action(context)
        self.do_game_end_highscore_action(context)
        self.do_game_end_score_action(context)

    def make_a_move(self, frame_buffer, context_frame):
        """
        Use the Neural Network to decide which actions to take in each step through Q-value estimates.
        """

        self.game_state.update_zoom_level(context_frame)
        kill_count = self.game_state.read_kill_count()
        health = self.game_state.update_health_counter(context_frame)  # self.update_health()
        #self.game_state.update_miss_counter(game_frame)

        global_state_count = 0
        frame_list = [x.frame for x in frame_buffer]
        #print("framelist:", frame_list)
        torch_state = torch.tensor(frame_list, dtype=torch.float32, device=self.replay_memory.device)
        #print(torch_state.shape)
        #for state in torch_state:
        #    for pixel_list in state:
        #        if any(pixel > 1 for pixel in pixel_list):
        #            print(state)
        #            raise ValueError('A very specific bad thing happened.')
        action = self.agent.act(torch_state)
        reward = self.agent.calculate_reward(kill_count)  # , health)
        self.game_state.episode_reward_total = self.game_state.episode_reward_total + reward
        self.game_state.kill_count = kill_count
        self.game_state.health = health
        if not self.episode_end:  # If health turns 0 we should only store that last state in memory. Ignore later states
            self.episode_end = True if self.game_state.health == 0 else False

            if self.val_mem.transitions.global_state_count < self.rainbow_arguments.evaluation_size:  # TODO: Clean this up. Replay memory does not count total number of states stored
                self.game_state.agent_mode = 'Building initial evaluation memory'
                global_state_count = self.val_mem.transitions.global_state_count
                # If we are still building validation memory
                self.val_mem.append(torch_state, None, None, self.episode_end)
            else:
                # When validation memory is full enough, use regular training replay memory
                global_state_count = self.replay_memory.transitions.global_state_count

                if self.agent.training == 'training':
                    self.game_state.agent_mode = 'Training'
                    if self.replay_memory.transitions.global_state_count % self.rainbow_arguments.replay_frequency == 0:
                        self.agent.reset_noise()  # Draw a new set of noisy weights

                    self.replay_memory.append(torch_state, action, reward, self.episode_end)

                    if self.replay_memory.transitions.global_state_count >= self.rainbow_arguments.learn_start:
                        self.replay_memory.priority_weight = min(self.replay_memory.priority_weight
                                                                 + self.priority_weight_increase,
                                                                 1)  # Anneal importance sampling weight β to 1
                elif self.agent.training == 'evaluation':
                    self.game_state.agent_mode = 'Evaluating training so far'
                    self.val_mem.append(torch_state, action, reward, self.episode_end)

        # For SerpentAI we need to select the corresponding key input mapping taken by the input_controller object
        # action is just the index of the move, first get the text meaning of the move, then the key object
        action_meaning = self.ACTION_MEANING_OFDP[action]
        #print("Action:", action, "Move:", action_meaning)
        buttons = self.input_mapping[action_meaning]

        #print(f"Clicking button: {buttons}")
        #test_time_start = time.time()


        #self.input_controller.handle_keys(key_collection=buttons) # This method does not seem to work
        for button in buttons:
            # self.input_controller.click(button=button)
            self.input_controller.tap_key(button)

        # test_time = time.time() - test_time_start

        if self.replay_memory.transitions.global_state_count >= self.rainbow_arguments.learn_start and \
                self.replay_memory.transitions.global_state_count % self.rainbow_arguments.replay_frequency == 0:
                # Train dqn at end of every few states
                print("Training agent", self.env_name)
                self.replay_memory.priority_weight = min(self.replay_memory.priority_weight +
                                                         self.priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                self.training_loss = self.agent.learn(self.replay_memory)
                print("Finished training agent", self.env_name)

        self.agent.printer.print_statistics(action=action,
                                            reward=reward,
                                            move_time=self.move_time,
                                            agent_training=self.agent.training,  # self.game_state.q_values,
                                            global_state_count=global_state_count,
                                            action_meaning_ofdp=self.ACTION_MEANING_OFDP,
                                            last_updated_net_time=self.last_updated_net_time,
                                            last_updated_net_episode=self.last_updated_net_episode)

    def end_episode(self):
        # Calculated survival time for current episode in seconds
        self.game_state.episode_time = time.time() - self.game_state.episode_start_time

        self.agent.printer.add_printer_head()

        # The memory address is not stable and some times contain another memory pointer
        # When kill count contains a pointer we should not use this episode
        #if self.game_state.kill_count < 1000:
            # Add the episode to the rest of replay memory and calculate statistics.
            # states, q_values, actions, rewards, end_episode = self.replay_memory.episode_memory.episode_end()

            #self.replay_memory.add_episode_too_memory(self.game_state.episode_time, self.agent.printer)

        self.agent.printer.flush()
        """
        if self.replay_memory.episode_count % 10 == 0:
            print("Saving replay memory checkpoint, please give me a minute...")
            self.replay_memory.store_memory_checkpoint()
            print("Finished storing replay memory for agent", self.env_name)

            # Train dqn at end of every few episodes
            print("Training agent", self.env_name)
            self.agent.learn(self.replay_memory)
            print("Finished training agent", self.env_name)
        """
        self.agent.reset_noise()  # Draw a new set of noisy weights

        if self.replay_memory.transitions.global_state_count >= self.rainbow_arguments.learn_start:
            if self.replay_memory.transitions.cur_episode % 200 == 0:
                print("Saving replay memory checkpoint, please give me a minute...")
                self.replay_memory.store_memory_checkpoint()
                print("Finished storing replay memory for agent", self.env_name)

            if self.replay_memory.transitions.cur_episode % self.rainbow_arguments.replay_frequency == 0\
                    and self.agent.training == 'training':
                # Train dqn at end of every few episodes
                print("Training agent", self.env_name)
                self.agent.learn(self.replay_memory)
                print("Finished training agent", self.env_name)

            if self.agent.training == 'evaluation':
                # Count if agent online net is set to evaluation mode
                self.evaluation_run_count = self.evaluation_run_count + 1

                if self.evaluation_run_count >= self.rainbow_arguments.evaluation_episodes \
                        and len(self.val_mem.episode_statistics.reward_last_10) >= 10:
                    avg_reward, avg_q = test(self.rainbow_arguments,
                                             range(self.replay_memory.transitions.global_state_count - self.evaluation_run_count,
                                                   self.replay_memory.transitions.global_state_count),
                                             self.agent, self.val_mem)
                    self.replay_memory.episode_statistics.update_eval(avg_reward, avg_q)
                    self.agent.train()  # Set DQN (online network) back to training mode

            if self.replay_memory.transitions.cur_episode % self.rainbow_arguments.evaluation_interval == 0:
                self.agent.eval()  # Set DQN (online network) to evaluation mode
                # val_mem stats are only used to evaluate reward over
                # self.rainbow_arguments.evaluation_episodes number of episodes
                # Reset this at the start of each evaluation run to track
                # the reward during evaluation for use in test.py
                self.val_mem.episode_statistics.reset_statistics()
                self.evaluation_run_count = 0

            # Update target network
            if self.replay_memory.transitions.cur_episode % self.rainbow_arguments.target_update == 0:
                self.last_updated_net_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.last_updated_net_episode = self.replay_memory.transitions.cur_episode
                self.agent.update_target_net()

        self.replay_memory.transitions.inc_episode()
        self.game_state.reset_game_state()

        # Perform garbage collection after each episode to avoid memory leakage over time
        #collected_garbage = gc.collect()
        #print("Garbage collector collected:", collected_garbage)

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
            self.episode_end = False
            self.survival_start_time = time.time()
            #print("Who needs skill. Let's play !")

            self.game_state.reset_game_state()
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
                    #pyautogui.mouseDown()
                    #pyautogui.mouseUp()

                self.username_entered = True

            print("Entering nickname, done !")
            self.username_entered = False

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="HS_OK"
            )

    def do_game_end_score_action(self, context):
        if context == "ofdp_game_end_score":
            self.end_episode()
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            time.sleep(3)

class tmpClassifier:
    def predict(self, frame):
        return "ofdp_game_paused"
