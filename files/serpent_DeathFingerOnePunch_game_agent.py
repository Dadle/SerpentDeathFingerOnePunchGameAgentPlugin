import time
import os
import gc
import psutil
import offshoot
from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

from .helpers.memory_manager import MemoryManager
from .helpers.game_state import GameState
from .helpers.image_plotter import ImagePlotter

from plugins.SerpentDeathFingerOnePunchGameAgentPlugin.files.ddqn import dqn_core as dqn

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
        self.downscale_img_size = (int(self.game.window_geometry['height']/10),
                                   int(self.game.window_geometry['width']/10),
                                   2)  # This means 1 greyscale image and 1 greyscale motion-trace image

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

        # Setup DQN network and load from checkpoint if available
        self.agent = dqn.Agent(action_list=self.ACTION_MEANING_OFDP,
                               state_shape=self.downscale_img_size,
                               env_name=self.env_name,
                               training=True,  # TODO ----------------> This is the switch between training and testing
                               checkpoint_base_dir_arg='checkpoints')


        # Convenient access to vital objects
        # Direct reference to the ANN in our agent for convenience
        self.model = self.agent.model
        # Direct reference to replay_memory for convenience
        self.replay_memory = self.agent.replay_memory


        self.game_state = GameState(training=self.agent.training,
                                    printer=self.agent.printer,
                                    memory_manager=self.memory_manager,
                                    downscale_img_size=self.downscale_img_size)
        self.agent.game_state = self.game_state
        self.agent.printer.game_state = self.game_state



        # render the output of the cnn layers for a state to see what features are learned
        #self.image_plotter.render_cnn_layer_outputs(self.game_state)

    def setup_context_classifier(self):
        context_classifier_path = f"{plugin_path}/SerpentDeathFingerOnePunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=self.window_dim)
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        return context_classifier

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        self.game_state.current_context = context
        self.game_state.not_playing_context_counter += 1

        #print(context)
        if (context is None or context in ["ofdp_playing", "ofdp_game"]) and self.game_state.health > 0:
            self.make_a_move(game_frame)
            self.game_state.not_playing_context_counter = 0
            return
        elif not all(x < 10000 for x in self.replay_memory.episode_memory.kill_count):
            # If kill_count gets really large during the episode we should ignore the episode
            # This happens if the memory address for kill count changes for this episode
            # and we should just ignore the episode
            self.agent.printer.print_error()

        else:
            # This is a hack to avoid runs being ended early due to
            # context classifier getting the wrong context while playing
            if self.game_state.not_playing_context_counter < 10:
                return

        self.do_splash_screen_action(context)
        self.do_main_menu_actions(context)
        self.do_mode_menu_action(context)
        self.do_survival_menu_action(context)
        self.do_survival_pre_game_action(context)
        self.do_game_paused_action(context)
        self.do_game_end_highscore_action(context)
        self.do_game_end_score_action(context)

    def make_a_move(self, game_frame):
        """
        Use the Neural Network to decide which actions to take in each step through Q-value estimates.
        """

        move_time_start = time.time()


        self.game_state.update_zoom_level(game_frame)
        self.game_state.kill_count = self.game_state.update_kill_count()
        self.game_state.health = self.game_state.update_health_counter(game_frame)  # self.update_health()
        #self.game_state.update_miss_counter(game_frame)

        action = self.agent.get_action(game_frame)

        # For SerpentAI we need to select the corresponding key input mapping taken by the input_controller object
        # action is just the index of the move, first get the text meaning of the move, then the key object
        action_meaning = self.ACTION_MEANING_OFDP[action]
        #print("MOVE:", action_meaning)
        buttons = self.input_mapping[action_meaning]

        #print(f"Clicking button: {buttons}")
        #test_time_start = time.time()


        #self.input_controller.handle_keys(key_collection=buttons) # This method does not seem to work
        for button in buttons:
            # self.input_controller.click(button=button)
            self.input_controller.tap_key(button)

        # test_time = time.time() - test_time_start

        move_time = time.time() - move_time_start
        self.agent.printer.print_statistics(move_time=move_time,
                                      q_values=self.game_state.q_values,
                                      action_meaning_ofdp=self.ACTION_MEANING_OFDP)

    def end_episode(self):
        # Calculated survival time for current episode in seconds
        self.game_state.episode_time = time.time() - self.game_state.episode_start_time

        self.agent.printer.add_printer_head()

        # The memory address is not stable and some times contain another memory pointer
        # When kill count contains a pointer we should not use this episode
        if self.game_state.kill_count < 1000:
            # Add the episode to the rest of replay memory and calculate statistics.
            # states, q_values, actions, rewards, end_episode = self.replay_memory.episode_memory.episode_end()
            self.replay_memory.add_episode_too_memory(self.game_state.episode_time, self.agent.printer)

        # TODO: use this to check if replay memory still looks correct
        # Print the last memory states for debugging
        """
        
        self.image_plotter.plot_images(self.replay_memory.states[self.replay_memory.num_used-9:],
                         self.replay_memory.kill_count[self.replay_memory.num_used-9:],
                         self.replay_memory.health[self.replay_memory.num_used-9:])
        """

        self.agent.printer.flush()

        if self.replay_memory.episode_statistics.num_episodes_completed % 10 == 0 \
                and self.replay_memory.episode_statistics.num_episodes_completed > 0\
                and self.agent.training:
            print("Saving replay memory checkpoint, please give me a minute...")
            self.replay_memory.store_memory_checkpoint()
            print("Finished storing replay memory for agent", self.env_name)

            # Train dqn at end of every few episodes
            self.agent.train_dqn()

        self.game_state.reset_game_state()

        # Perform garbage collection after each episode to avoid memory leakage over time
        collected_garbage = gc.collect()
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
