import numpy as np
import os
import glob
import collections
import pickle
from ..helpers.terminal_printer import TerminalPrinter

class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural Network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, size, num_actions, state_shape, env_name, checkpoint_dir,
                 discount_factor=0.8, error_threshold=0.1):
        """

        :param size:
            Capacity of the replay-memory. This is the number of states.
        :param num_actions:
            Number of possible actions in the game-environment.
        :param state_shape:
            Shape of each state object that will be added to the replay memory
        :param env_name:
            Name of the game agent. Used as part of filenames during checkpointing.
        :param checkpoint_dir:
            Path to the location where checkpoint files should be stored.
        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        self.checkpoint_path = checkpoint_dir

        # Name of this game agent, only used for checkpointing and naming files on disk
        self.env_name = env_name

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = error_threshold

        # Initialize a first episode memory
        self.episode_memory = self.EpisodeReplayMemory(self.discount_factor, self.error_threshold)

        # Object keeping track of statistics across episodes for printing
        self.episode_statistics = self.EpisodeStatistics()

        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Number of kills made since last end_episode for each of the states in the memory
        self.kill_count = np.zeros(shape=size, dtype=np.float)

        # Number of misses made since last end_episode for each of the states in the memory
        self.miss_count = np.zeros(shape=size, dtype=np.float)

        # number of health points for each state
        self.health = np.zeros(shape=size, dtype=np.bool)

        # Whether the episode had ended (aka. game over) in each state.
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        # Estimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # The number of times this object has been stored to file
        self.store_count = 0

    def store_memory_checkpoint(self):
        """Store Replay Memory object to file for checkpointing.
        The filename alternates between 1 and 2 in case something goes wrong while writing a checkpoint"""
        self.store_count += 1
        checkpoint_file_path = os.path.join(self.checkpoint_path, 'checkpoint_replay_memory_' + self.env_name +
                                            '_' + str(self.store_count % 2) + '.pkl')
        #checkpoint_file_path = 'checkpoint_replay_memory_' + self.env_name + '_' + str(self.store_count % 2) + '.pkl'
        #print(os.getcwd())
        #print(checkpoint_file_path)
        with open(checkpoint_file_path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_memory_checkpoint(env_name, checkpoint_dir):
        """Read the newest Replay Memory object file from disk and return the contained object"""
        # os.chdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, 'checkpoint_replay_memory_' + env_name + '_*.pkl')
        print(os.getcwd())
        #path = os.path.join(os.getcwd(), 'checkpoint_replay_memory_' + env_name + '_*.pkl')

        # Get a list of all files under the input path in sorted oder from newest to oldest.
        # Input patch can contain any filter accepted by glob.glob such as * for wildchard characters
        files = list(reversed(sorted(glob.glob(path), key=os.path.getmtime)))
        print("Found these checkpoint files:", files)
        # Try to read the file. The file could be corrupt and if so we should read the second newest file
        for file in files:
            #print("Trying to load:", file)
            try:
                with open(file, 'rb') as memory_file:
                    return pickle.load(memory_file)
            except Exception as e:
                #print("Got exception when loading replay memory")
                #print(e)
                continue
        #print("Finished trying to load replay memory")
        return None

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0
        self.episode_memory = self.EpisodeReplayMemory(self.discount_factor, self.error_threshold)
        self.episode_statistics = self.EpisodeStatistics()

    def reset_episode(self):
        """Reset the episode replay-memory so it is ready for a new episode."""
        self.episode_memory = self.EpisodeReplayMemory(self.discount_factor, self.error_threshold)

    def add_episode_too_memory(self, episode_time_seconds, printer):
        """
        Adds the completed current episode memory to the replay-memory as training material
        """
        if len(self.episode_memory.kill_count) > 0:
            # With current implementation memory region for kill count is hard coded.
            # This region is some times swithced for short periods of time.
            # This code skips episodes where the memory region is wrong
            if self.episode_memory.kill_count[-1] > 10000:
                print("Kill count memory region is wrong. Not adding this episode to memory!")
                print("Memory region should reset itself in a few rounds restoring normal operation. "
                      "\nPlease stand by.")
                self.reset_episode()
                return

        # If episode checks out OK it is added to the rest of the replay memory
        # print("Adding episode to replay memory")

        self.episode_memory.end_episode_updates(printer)
        self.episode_statistics.update_statistics(self.episode_memory, episode_time_seconds)

        # While adding episode to replay memory we need to calculate the rewards
        for index in range(len(self.episode_memory.states)):
            if self.num_used >= self.size:
                # If replay memory is full, remove the oldest memory and add the newest one at the end
                #print("shape of memory after pop", self.states[1:].shape)
                #print("shape of episode state", np.array([self.episode_memory.states[index]]).shape)
                self.states = np.concatenate([self.states[1:], [self.episode_memory.states[index]]])
                self.q_values = np.concatenate([self.q_values[1:], [self.episode_memory.q_values[index]]])
                self.actions = np.concatenate([self.actions[1:], [self.episode_memory.actions[index]]])
                self.rewards = np.concatenate([self.rewards[1:], [self.episode_memory.rewards[index]]])
                self.kill_count = np.concatenate([self.kill_count[1:], [self.episode_memory.kill_count[index]]])
                self.miss_count = np.concatenate([self.miss_count[1:], [self.episode_memory.miss_count[index]]])
                self.health = np.concatenate([self.health[1:], [self.episode_memory.health[index]]])
                self.end_episode = np.concatenate([self.end_episode[1:], [self.episode_memory.end_episode[index]]])
                self.estimation_errors = np.concatenate([self.estimation_errors[1:],
                                                         [self.episode_memory.estimation_errors[index]]])
                break
            else:
                # If there is still room in replay memory, add the episode at the end of the currently used memory
                k = self.num_used

                # Increase the number of used elements in the replay-memory.
                self.num_used += 1
                self.states[k] = self.episode_memory.states[index]
                self.q_values[k] = self.episode_memory.q_values[index]
                self.actions[k] = self.episode_memory.actions[index]
                self.rewards[k] = self.episode_memory.rewards[index]
                self.kill_count[k] = self.episode_memory.kill_count[index]
                self.miss_count[k] = self.episode_memory.miss_count[index]
                self.health[k] = self.episode_memory.health[index]
                self.end_episode[k] = self.episode_memory.end_episode[index]
                self.estimation_errors[k] = self.episode_memory.estimation_errors[index]

        self.reset_episode()

    def add(self, state, q_values, action, reward, kill_count, miss_count, health, end_episode):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.
        :param q_values:
            The estimated Q-values for the state.
        :param action:
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        :param kill_count:
            Number of kills made since last end_episode flag
        :param health:
            Number of health points for player. Int between 0 and 10
        :param end_episode:
            Boolean whether the agent has lost all lives aka. game over
            aka. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elements in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.kill_count[k] = kill_count
            self.miss_count[k] = miss_count
            self.health[k] = health
            self.end_episode[k] = end_episode

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            self.rewards[k] = np.clip(reward, -1.0, 1.0)
        else:
            # delete the oldest memory state and add the new one at the end as usual
            # Store all the values in the replay-memory.
            self.states = np.concatenate([self.states[1:], [state]])
            self.q_values = np.concatenate([self.q_values[1:], [q_values]])
            self.actions = np.concatenate([self.actions[1:], [action]])
            self.kill_count = np.concatenate([self.kill_count[1:], [kill_count]])
            self.miss_count = np.concatenate([self.miss_count[1:], [miss_count]])
            self.health = np.concatenate([self.health[1:], [health]])
            self.end_episode = np.concatenate([self.end_episode[1:], [end_episode]])

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            self.rewards = np.concatenate([self.rewards[1:], [np.clip(reward, -1.0, 1.0)]])

    def update_all_q_values(self, printer):
        """
        Update all Q-values in the replay-memory.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used - 1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards
            end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward[k]
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward[k+1] + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

        self.print_statistics(printer)

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.
        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.

        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err < self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        if isinstance(self.idx_err_hi, np.ndarray):
            try:
                prob_err_hi = self.idx_err_hi.size / self.num_used
            except Exception as e:
                print(str(e))
                print("idx_err_hi: ", self.idx_err_hi, "of type", type(self.idx_err_hi))
        else:

            prob_err_hi = self.idx_err_hi / self.num_used
        try:
            prob_err_hi = max(prob_err_hi, 0.5)
        except Exception as e:
            print(str(e))
            print("prob_err_hi: ", prob_err_hi, "of type", type(prob_err_hi))

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.
        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        idx_lo = []
        idx_hi = []
        try:
            # Random index of states and Q-values in the replay-memory.
            # These have LOW estimation errors for the Q-values.
            idx_lo = np.random.choice(self.idx_err_lo,
                                      size=self.num_samples_err_lo,
                                      replace=False)
        except ValueError:
            print("*****selecting low error batch failed*****")

        try:
            # Random index of states and Q-values in the replay-memory.
            # These have HIGH estimation errors for the Q-values.
            idx_hi = np.random.choice(self.idx_err_hi,
                                      size=self.num_samples_err_hi,
                                      replace=False)
        except ValueError:
            print("*****selecting high error batch failed*****")

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi)).astype(int)

        # Get the batches of states and Q-values.
        try:
            states_batch = self.states[idx]
            q_values_batch = self.q_values[idx]
        except IndexError as e:
            print("Retrieving states for training failed")
            print("IDX contains:", idx)
            print("EXCEPTION:", str(e))

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.

        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.
        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.
        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def print_statistics(self, printer):
        """Print statistics for the contents of the replay-memory."""

        printer.add("Replay-memory statistics:")

        episode_lenghts = []
        episode_rewards = []

        episode_start_idx = 0
        for idx, flag in enumerate(self.end_episode):
            if flag:
                episode_lenghts.append(idx-episode_start_idx)
                episode_rewards.append(sum(self.rewards[episode_start_idx:idx]))
                episode_start_idx = idx+1

        printer.add(f"Number of episodes: {len(episode_lenghts)}")
        printer.add(f"last end episode flag: {episode_start_idx-1}")

        # Print statistics for the episodes that are contained in replay memory.
        msg = "\tEpisode length are min: {0:5.2f}, mean: {1:5.2f}, max: {2:5.2f}"
        printer.add(msg.format(np.min(episode_lenghts),
                               np.mean(episode_lenghts),
                               np.max(episode_lenghts)))

        # Print statistics for the episodes that are contained in replay memory.
        msg = "\tEpisode rewards are min: {0:5.2f}, mean: {1:5.2f}, max: {2:5.2f}"
        printer.add(msg.format(np.min(episode_rewards),
                               np.mean(episode_rewards),
                               np.max(episode_rewards)))


        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        printer.add(msg.format(np.min(self.q_values_old),
                               np.mean(self.q_values_old),
                               np.max(self.q_values_old)))

        """
        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        printer.add(msg.format(np.min(self.q_values),
                               np.mean(self.q_values),
                               np.max(self.q_values)))
        
        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        printer.add(msg.format(np.min(q_dif),
                               np.mean(q_dif),
                               np.max(q_dif)))
        """

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        printer.add(msg.format(self.error_threshold, err_count,
                               self.num_used, err_count / self.num_used))

    class EpisodeReplayMemory:
        """
        Episode replay memory stores the states for an episode until we can calculate rewards and add the memory
        to the historic replay memory and use it for training
        """
        def __init__(self, discount_factor, error_threshold):
            """
            The initialization is the same as for ReplayMemory,
            but uses dynamic arrays since we do not know the length of an episode

            :param discount_factor:
                Discount-factor used for updating Q-values.
            """

            # Array for the previous states of the game-environment.
            self.states = []

            # Array for the Q-values corresponding to the states.
            self.q_values = []

            # Array for the Q-values before being updated.
            # This is used to compare the Q-values before and after the update.
            self.q_values_old = []

            # Actions taken for each of the states in memory
            self.actions = []

            # Rewards observed for each of the states in memory
            self.rewards = []

            # kill_count observed for each of the states in memory
            self.kill_count = []

            # Number of times player has missed for each state in memory
            self.miss_count = []

            # Number of health points for player for each state in memory
            self.health = []

            # Whether the episode had ended (aka. game over) in each state.
            self.end_episode = []

            # Estimation errors for the Q-values. This is used to balance
            # the sampling of batches for training the Neural Network,
            # so we get a balanced combination of states with high and low
            # estimation errors for their Q-values.
            self.estimation_errors = []

            # Discount-factor used for updating Q-values.
            self.discount_factor = discount_factor

            # Threshold for splitting between low and high estimation errors.
            self.error_threshold = error_threshold

        def reset(self):
            """Reset the episode replay-memory so it is empty."""

            # Array for the previous states of the game-environment.
            self.states = []

            # Array for the Q-values corresponding to the states.
            self.q_values = []

            # Array for the Q-values before being updated.
            # This is used to compare the Q-values before and after the update.
            self.q_values_old = []

            # Actions taken for each state in memory.
            self.actions = []

            # Rewards observed for each state in memory.
            self.rewards = []

            # kill_count observed for each state in memory.
            self.kill_count = []

            # Number of times player has missed for each state in memory
            self.miss_count = []

            # Number of health points for player for each state in memory
            self.health = []

            # Whether the episode had ended (aka. game over) in each state.
            self.end_episode = []

            # Estimation errors for the Q-values. This is used to balance
            # the sampling of batches for training the Neural Network,
            # so we get a balanced combination of states with high and low
            # estimation errors for their Q-values.
            self.estimation_errors = []

            self.num_used = 0

        def add(self, state, q_values, action, reward, kill_count, miss_count, health):
            """
            Add an observed state from the game-environment, along with the
            estimated Q-values, action taken, etc.

            :param state:
                Current state of the game-environment.
                This is the output of the MotionTracer-class.
            :param q_values:
                The estimated Q-values for the state.
            :param action:
                The action taken by the agent in this state of the game.
            :param reward:
                The reward given to the game agent for it's actions
            :param kill_count:
                Number of kills made so far this run
            :param miss_count:
                Number of times player has missed so far this run
            :param health:
                Number of health points for player. Int between 0 and 10
            """

            # set size of episode memory
            self.num_used = len(self.states)

            # Store all the values in the replay-memory.
            self.states.append(state)
            self.q_values.append(q_values)
            self.actions.append(action)
            self.rewards.append(reward)
            self.kill_count.append(kill_count)
            self.miss_count.append(miss_count)
            self.health.append(health)
            # Boolean whether the agent has lost all lives aka. game over aka. end of episode.
            # The last flag in each episode memory is set to True while
            # adding episode memory to the rest of replay memory
            self.end_episode.append(False)

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            #self.rewards[k] = np.clip(reward, -1.0, 1.0)

        def update_episode_q_values(self):
            """
            Update all Q-values in the replay-memory.

            When states and Q-values are added to the replay-memory, the
            Q-values have been estimated by the Neural Network. But we now
            have more data available that we can use to improve the estimated
            Q-values, because we now know which actions were taken and the
            observed rewards. We sweep backwards through the entire replay-memory
            to use the observed data to improve the estimated Q-values.
            """

            # Copy old Q-values so we can print their statistics later.
            # Note that the contents of the arrays are copied.
            self.q_values_old[:] = self.q_values[:]

            # Now that we know how long the entire episode is we can initialize the estimation errors array
            self.estimation_errors = np.zeros(shape=len(self.states), dtype=np.float)

            # Process the replay-memory backwards and update the Q-values.
            # This loop could be implemented entirely in NumPy for higher speed,
            # but it is probably only a small fraction of the overall time usage,
            # and it is much easier to understand when implemented like this.
            for k in reversed(range(len(self.states))):
                # Get the data for the k'th state in the replay-memory.
                action = self.actions[k]
                reward = self.rewards[k]
                end_episode = self.end_episode[k]

                # Calculate the Q-value for the action that was taken in this state.
                if end_episode:
                    # If the agent lost a life or it was game over / end of episode,
                    # then the value of taking the given action is just the reward
                    # that was observed in this single step. This is because the
                    # Q-value is defined as the discounted value of all future game
                    # steps in a single life of the agent. When the life has ended,
                    # there will be no future steps.
                    action_value = reward
                else:
                    # Otherwise the value of taking the action is the reward that
                    # we have observed plus the discounted value of future rewards
                    # from continuing the game. We use the estimated Q-values for
                    # the following state and take the maximum, because we will
                    # generally take the action that has the highest Q-value.
                    action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

                # Error of the Q-value that was estimated using the Neural Network.
                self.estimation_errors[k] = abs(action_value - self.q_values[k][action])

                # Update the Q-value with the better estimate.
                self.q_values[k][action] = action_value

        def end_episode_updates(self, printer):
            # printer.add(f"Number of states: {len(self.states)}")
            if len(self.states) > 0:
                # At the end of each episode go through all q_values for the episode and set
                # the correct action values based on experienced rewards.
                # This will take each reward and punishment and spread it backwards in time to give
                # an estimated value for each action taken known as an action value.

                self.end_episode[-1] = True
                # printer.add(f"episode end flag: {self.end_episode[-1]}")

                # TODO: Consider if it is possible to get away with only calculating action values once
                # Having growing Q values over time seems strange
                # We could focus on which action is most likely the best one for this state (likelihood)
                # And not what predicted reward we could get from this action (prediction).
                # Prediction depends on all future actions as well to calculate cumulative reward
                self.update_episode_q_values()
                self.print_episode_q_statistics(printer=printer)



        @staticmethod
        def reward(state_number, episode_length):
            """
                        NOT IN USE IN THIS MODULE - REWARD IS PROVIDED EXTERNALLY

                        When the episode ends we know the final score and can run the
                        episode again to calculate the reward per state

                        :param: state_number:
                            Index of the state for the episode
                        :param: episode_length:
                            number of states in the episode
                        :return: reward:
                            Reward for the state given as state_number in this episode
                        """
            # print(f"CALCULATED REWARD: {(1 - (1 / episode_length)) ** state_number}")
            return (1 - (1 / episode_length)) ** state_number

        def print_episode_q_statistics(self, printer):
            """Print statistics for the episode in the episode replay-memory."""

            printer.add("Episode replay-memory statistics:")

            # Print episode length
            printer.add(f"\tEpisode lasted {len(self.states)} states resulting in total reward of {sum(self.rewards)}")

            # Print statistics for the Q-values before they were updated
            # in update_all_q_values().
            msg = "\tQ-values Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
            printer.add(msg.format(np.min(self.q_values_old),
                                   np.mean(self.q_values_old),
                                   np.max(self.q_values_old)))

            """
            # Print statistics for the Q-values after they were updated
            # in update_all_q_values().
            msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
            printer.add(msg.format(np.min(self.q_values),
                                   np.mean(self.q_values),
                                   np.max(self.q_values)))

            # Print statistics for the difference in Q-values before and
            # after the update in update_all_q_values().
            q_dif = np.array(self.q_values) - np.array(self.q_values_old)
            msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
            printer.add(msg.format(np.min(q_dif),
                                   np.mean(q_dif),
                                   np.max(q_dif)))
            """

            # Print statistics for the number of large estimation errors.
            # Don't use the estimation error for the last state in the memory,
            # because its Q-values have not been updated.
            err = self.estimation_errors[:-1]
            err_count = np.count_nonzero(err > self.error_threshold)
            if self.num_used > 0:
                err_pct = err_count / self.num_used
            else:
                err_pct = 0
            msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
            printer.add(msg.format(self.error_threshold, err_count,
                                   self.num_used, err_pct))

    class EpisodeStatistics:
        """
        Statistics for printing the status of the game agent and its achievements
        """

        def __init__(self):
            self.record_episode = 0
            self.record_episode_time = 0
            self.record_kill_count = 0

            self.time_last_10 = collections.deque(list(), maxlen=10)
            self.time_last_100 = collections.deque(list(), maxlen=100)
            self.time_last_1000 = collections.deque(list(), maxlen=1000)
            self.average_time_10 = 0
            self.average_time_100 = 0
            self.average_time_1000 = 0

            self.kill_count_last_10 = collections.deque(list(), maxlen=10)
            self.kill_count_last_100 = collections.deque(list(), maxlen=100)
            self.kill_count_last_1000 = collections.deque(list(), maxlen=1000)
            self.average_kill_count_10 = 0
            self.average_kill_count_100 = 0
            self.average_kill_count_1000 = 0

            self.num_episodes_completed = 1
            self.last_episode_time = 0
            self.last_episode_kill_count = 0

        def update_statistics(self, episode_memory, episode_time_seconds):
            # Catch if this is called before any episode has been run so it does not crash
            if len(episode_memory.states) == 0:
                return

            self.num_episodes_completed += 1
            self.last_episode_time = episode_time_seconds
            self.last_episode_kill_count = episode_memory.kill_count[-1]

            if self.record_episode_time < self.last_episode_time:
                self.record_episode_time = self.last_episode_time
                self.record_episode = self.num_episodes_completed
                self.record_kill_count = episode_memory.kill_count[-1]

            self.time_last_10.appendleft(self.last_episode_time)
            self.time_last_100.appendleft(self.last_episode_time)
            self.time_last_1000.appendleft(self.last_episode_time)
            self.average_time_10 = float(np.mean(self.time_last_10))
            self.average_time_100 = float(np.mean(self.time_last_100))
            self.average_time_1000 = float(np.mean(self.time_last_1000))

            self.kill_count_last_10.appendleft(episode_memory.kill_count[-1])
            self.kill_count_last_100.appendleft(episode_memory.kill_count[-1])
            self.kill_count_last_1000.appendleft(episode_memory.kill_count[-1])
            self.average_kill_count_10 = float(np.mean(self.kill_count_last_10))
            self.average_kill_count_100 = float(np.mean(self.kill_count_last_100))
            self.average_kill_count_1000 = float(np.mean(self.kill_count_last_1000))

