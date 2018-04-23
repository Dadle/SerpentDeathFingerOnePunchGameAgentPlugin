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
from serpent import ocr

from plugins.SerpentDeathFingerOnePunchGameAgentPlugin.files.ddqn import dqn_core as dqn

plugin_path = offshoot.config["file_paths"]["plugins"]

class SerpentDeathFingerOnePunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        print("Game agent initiating")
        super().__init__(**kwargs)

        self.process = psutil.Process(os.getpid())

        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)
        self.memory_timeframe = 6  #at 2 FPS means 3 seconds of history

        #self.player_model = KerasDeepPlayer(time_dim=(self.memory_timeframe,),
        #                                    game_frame_dim=self.window_dim)

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

        # Complete action meaning from OpenAI gym
        self.ACTION_MEANING_GYM = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN",
            6: "UPRIGHT",
            7: "UPLEFT",
            8: "DOWNRIGHT",
            9: "DOWNLEFT",
            10: "UPFIRE",
            11: "RIGHTFIRE",
            12: "LEFTFIRE",
            13: "DOWNFIRE",
            14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE",
            17: "DOWNLEFTFIRE",
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

        self.new_episode = True

        print("Game agent finished initiating")

    def setup_play(self):
        self.kill_count = 0
        self.reward_episode = -2.0
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

    def setup_ddqn(self):
        env_name = 'ofdp'
        dqn.checkpoint_base_dir = 'checkpoints_tutorial16/'
        dqn.update_paths(env_name=env_name)

        # Setup DQN network and load from checkpoint if available
        self.agent = dqn.Agent(action_list=self.ACTION_MEANING_OFDP,
                                 training=True,
                                 render=True,
                                 use_logging=False)

        # Direct reference to the ANN in our agent for convenience
        self.model = self.agent.model

        # Direct reference to replay_memory for convenience
        self.replay_memory = self.agent.replay_memory

        self.episode_start_time = datetime.now()

    def handle_play(self, game_frame):
        #gc.disable()
        self.context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        #print(self.context)
        if self.context is None or self.context == "ofdp_game":
            #print("FIGHT!")
            self.game_state["alive"].appendleft(1)
            #print("There's nothing there... Waiting...")
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

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

    # Does not work unfortunately. Transform just hangs
    def _ocr_update_kill_count(self, game_frame):
        kill_count_region = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions["KILL_COUNT"]
        )

        print("Kill count region extracted:", kill_count_region.shape)

        kill_count_observed = ocr.perform_ocr(kill_count_region)

        print("Read kill count:", kill_count_observed)

        if kill_count_observed != "":
            if kill_count_observed > self.kill_count and kill_count_observed < self.kill_count + 5:
                self.kill_count = kill_count_observed

        print("Current number of kills:", self.kill_count)

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

        # Take a step in the game-environment using the given action.
        # Note that in OpenAI Gym, the step-function actually repeats the
        # action between 2 and 4 time-steps for Atari games, with the number
        # chosen at random.

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
        self.replay_memory.add_episode_too_memory(self.model.count_episodes)
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
                                      reward_episode=self.reward_episode,
                                      reward_mean=reward_mean)

            # Print reward to screen.
            msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
            print(msg.format(count_episodes, count_states, self.epsilon,
                             self.reward_episode, reward_mean))

        episode_time = datetime.now() - self.episode_start_time

        self.printer.add("")
        self.printer.add("Capgemini Intelligent Automation - Death Finger One Punch agent")
        self.printer.add("Reinforcement Learning: Training a DQN Agent")
        self.printer.add("")
        self.printer.add(f"Episode Started At: {self.episode_start_time.strftime('%Y-%m-%d %H:%M')}")
        self.printer.add(f"Episode finished At: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.printer.add(f"Episode lasted: {episode_time.seconds // 3600} hours, "
                         f"{(episode_time.seconds // 60) % 60} minutes, "
                         f"{episode_time.seconds % 60} seconds")
        self.printer.add(f"Completed episodes: {self.model.get_count_episodes()}")

        self.printer.add("")

        self.printer.flush()

        # Train dqn at end of each episode
        self.train_dqn()
        self.reset_game_state()

    def print_statistics(self):
        run_time = datetime.now() - self.started_at
        episode_time = datetime.now() - self.episode_start_time
        #serpent.utilities.clear_terminal()

        self.printer.add("")
        self.printer.add("Capgemini Intelligent Automation - Death Finger One Punch agent")
        self.printer.add("Reinforcement Learning: Training a DQN Agent")
        self.printer.add("")

        self.printer.add("\033c" + f"SESSION RUN TIME: "
                                   f"{run_time.days} days, "
                                   f"{run_time.seconds // 3600} hours, "
                                   f"{(run_time.seconds // 60) % 60} minutes, "
                                   f"{run_time.seconds % 60} seconds")
        self.printer.add("")
        self.printer.add(f"CURRENT EPISODE: #{self.model.get_count_episodes()}")
        self.printer.add(f"Episode FPS: {round(episode_time.seconds/len(self.replay_memory.episode_memory.states), 2)}")
        self.printer.add(f"Episode clock time: "
                         f"{episode_time.seconds // 3600} hours, "
                         f"{(episode_time.seconds // 60) % 60} minutes, "
                         f"{episode_time.seconds % 60} seconds")
        self.printer.add(f"Episode states processed: {len(self.replay_memory.episode_memory.states)}")
        self.printer.add("")

        self.printer.add(f"Average Rewards (Last 10 Runs): "
                         f"{round(self.replay_memory.episode_statistics.average_reward_10, 2)}")
        self.printer.add(f"Average Rewards (Last 100 Runs): "
                         f"{round(self.replay_memory.episode_statistics.average_reward_100, 2)}")
        self.printer.add(f"Average Rewards (Last 1000 Runs): "
                         f"{round(self.replay_memory.episode_statistics.average_reward_1000, 2)}")

        self.printer.add("")

        #self.printer.add(f"Reading context: {self.context}")

        self.printer.add(f"NEURAL NETWORK MOVE: "
                         f"{self.ACTION_MEANING_OFDP[self.replay_memory.episode_memory.actions[-1]]}")
        self.printer.add(f"States processed: {len(self.replay_memory.episode_memory.states)} "
                         f"\nq_values: {self.replay_memory.episode_memory.q_values[-1]} "
                         f"\nAction: {self.replay_memory.episode_memory.actions[-1]}")

        self.printer.add("")

        if self.replay_memory.num_used > 0:
            self.printer.add("")
            self.printer.add(f"CURR RUN TIME ALIVE: {len(self.replay_memory.episode_memory.states)/2} seconds")
            self.printer.add(f"LAST RUN TIME ALIVE: {self.replay_memory.episode_statistics.last_episode_time} seconds")

        self.printer.add("")
        self.printer.add(f"RECORD TIME ALIVE: "
                         f"{self.replay_memory.episode_statistics.record_episode_length} seconds "
                         f"(Run {self.replay_memory.episode_statistics.record_episode}")
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
        use_fraction = self.agent.replay_fraction.get_value(iteration=count_states)

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


    def old_DDQN_code(self):
        ###
        # ddqn execution code
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

    def train_ddqn(self):
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
                if self.dqn_movement.type == "ddqn":
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