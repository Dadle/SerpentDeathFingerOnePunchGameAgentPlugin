import random
import collections
from collections import namedtuple
import torch
import os
import pickle
import glob
import numpy as np



Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros((139, 139, 3), dtype=torch.uint8), None, 0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
        self.data = [None] * size  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)
        self.cur_episode = 0
        self.global_state_count = 0

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

    def inc_episode(self):
        self.cur_episode += 1

    def inc_global_state_count(self):
        self.global_state_count += 1


class ReplayMemory:
    def __init__(self, args, capacity, memory_type):
        self.env_name = args.env_name
        self.memory_type = memory_type
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.episode_count = 0
        self.transitions = self.load_memory_checkpoint('memory', memory_type, self.env_name, 'checkpoints')
        if self.transitions is None:
            self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.episode_statistics = self.load_memory_checkpoint('statistics', memory_type, self.env_name, 'checkpoints')
        if self.episode_statistics is None:
            self.episode_statistics = EpisodeStatistics()
        self.game_state = None  # Set after creation

        self.store_count = 0
        self.checkpoint_path = args.checkpoint_path

    def store_memory_checkpoint(self):
        """Store Replay Memory object to file for checkpointing.
                The filename alternates between 1 and 2 in case something goes wrong while writing a checkpoint"""
        self.store_count += 1
        trans_checkpoint_file_path = os.path.join(self.checkpoint_path, 'checkpoint_' + self.memory_type +
                                                  '_replay_memory_' + self.env_name +
                                                  '_' + str(self.store_count % 2) + '.pkl')
        stats_checkpoint_file_path = os.path.join(self.checkpoint_path, 'checkpoint_' + self.memory_type +
                                                  '_replay_statistics_' + self.env_name +
                                                  '_' + str(self.store_count % 2) + '.pkl')
        # checkpoint_file_path = 'checkpoint_replay_memory_' + self.env_name + '_' + str(self.store_count % 2) + '.pkl'
        # print(os.getcwd())
        # print(checkpoint_file_path)
        with open(trans_checkpoint_file_path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.transitions, output, pickle.HIGHEST_PROTOCOL)

        with open(stats_checkpoint_file_path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.episode_statistics, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_memory_checkpoint(checkpoint_type, memory_type, env_name, checkpoint_dir):
        """Read the newest Replay Memory object file from disk and return the contained object"""
        # os.chdir(checkpoint_dir)
        path = os.path.join(checkpoint_dir, 'checkpoint_' + memory_type + '_replay_' +
                            checkpoint_type + '_' + env_name + '_*.pkl')
        #print(os.getcwd())
        # path = os.path.join(os.getcwd(), 'checkpoint_replay_memory_' + env_name + '_*.pkl')

        # Get a list of all files under the input path in sorted oder from newest to oldest.
        # Input patch can contain any filter accepted by glob.glob such as * for wildchard characters
        files = list(reversed(sorted(glob.glob(path), key=os.path.getmtime)))
        print(path)
        print("Found these checkpoint files:", files)
        # Try to read the file. The file could be corrupt and if so we should read the second newest file
        for file in files:
            # print("Trying to load:", file)
            try:
                with open(file, 'rb') as memory_file:
                    memory_object = pickle.load(memory_file)
                    print("Successfully loaded memory file", file)
                    return memory_object
            except Exception as e:
                print("Got exception when loading memory file", file)
                print(e)
                continue
        # print("Finished trying to load replay memory")
        return None

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
        if terminal:
            self.t = 0
            self.episode_count = self.episode_count + 1
            self.episode_statistics.update_statistics(self.episode_count,
                                                      self.game_state.kill_count,
                                                      self.game_state.episode_time,
                                                      self.game_state.episode_reward_total)
        else:
            self.t += 1  # Start new episodes with t = 0
        self.transitions.inc_global_state_count()
        #print('append')

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = [None] * (self.history + self.n)
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                #print('_get_transition t+1')
                transition[t] = blank_trans  # If future frame has timestep 0
            else:
                #print('_get_transition t+1 else')
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
                #print('_get_transition not blank non terminal')
            else:
                #print('_get_transition blank')
                transition[t] = blank_trans  # If prev (next) frame is terminal

        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        """print("state list len: ", len(transition), type(transition))
        print("transition 0: ", transition[0], type(transition[0]))
        print("state 0 shape: ", transition[0].state.shape, type(transition[0].state), type(transition[0].state[0]))
        print("state 1 shape: ", transition[1].state.shape, type(transition[1].state), type(transition[0].state[1]))
        print("state 2 shape: ", transition[2].state.shape, type(transition[2].state), type(transition[0].state[2]))
        print("state 3 shape: ", transition[3].state.shape, type(transition[3].state), type(transition[0].state[3]))
        print("state: ", transition)
        """
        #print('_get_sample_from_segment')
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state.to(dtype=torch.float32, device=self.device) for trans in transition[:self.history]]).to(dtype=torch.float32, device=self.device).div_(255)
        #print("state shape: ", state.shape, type(state))
        #print("state: ", state)

        transition_states = [trans.state for trans in transition[self.n:self.n + self.history]]
        state_stack = torch.stack(transition_states)
        next_state = state_stack.to(dtype=torch.float32, device=self.device).div_(255)
        #next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(dtype=torch.float32, device=self.device).div_(255)
        #print("next_state shape: ", next_state.shape, type(next_state))
        #print("next_state: ", next_state)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = torch.tensor(probs, dtype=torch.float32, device=self.device) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = weights / weights.max()   # Normalise by max importance-sampling weight from batch
        #print('sample')
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities.pow_(self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state


class EpisodeStatistics:
    """
    Statistics for printing the status of the game agent and its achievements
    """

    def __init__(self):
        self.record_episode = 0
        self.record_episode_time = 0
        self.record_kill_count = 0

        self.reward_last_10 = collections.deque(list(), maxlen=10)
        self.reward_last_100 = collections.deque(list(), maxlen=100)
        self.reward_last_1000 = collections.deque(list(), maxlen=1000)
        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

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

        self.eval_reward_last = 0
        self.eval_avg_q_last = 0
        self.eval_reward_record = 0
        self.eval_avg_q_record = 0

    def update_statistics(self, episode_count, kill_count, episode_time_seconds, episeode_reward):
        # Catch if this is called before any episode has been run
        if episode_count == 0:
            return

        self.num_episodes_completed = episode_count
        self.last_episode_time = episode_time_seconds
        self.last_episode_kill_count = kill_count

        if self.record_episode_time < self.last_episode_time:
            self.record_episode_time = self.last_episode_time
            self.record_episode = self.num_episodes_completed
            self.record_kill_count = kill_count

        self.reward_last_10.appendleft(episeode_reward)
        self.reward_last_100.appendleft(episeode_reward)
        self.reward_last_1000.appendleft(episeode_reward)
        self.average_reward_10 = float(np.mean(self.reward_last_10))
        self.average_reward_100 = float(np.mean(self.reward_last_100))
        self.average_reward_1000 = float(np.mean(self.reward_last_1000))

        self.time_last_10.appendleft(self.last_episode_time)
        self.time_last_100.appendleft(self.last_episode_time)
        self.time_last_1000.appendleft(self.last_episode_time)
        self.average_time_10 = float(np.mean(self.time_last_10))
        self.average_time_100 = float(np.mean(self.time_last_100))
        self.average_time_1000 = float(np.mean(self.time_last_1000))

        self.kill_count_last_10.appendleft(kill_count)
        self.kill_count_last_100.appendleft(kill_count)
        self.kill_count_last_1000.appendleft(kill_count)
        self.average_kill_count_10 = float(np.mean(self.kill_count_last_10))
        self.average_kill_count_100 = float(np.mean(self.kill_count_last_100))
        self.average_kill_count_1000 = float(np.mean(self.kill_count_last_1000))

    def update_eval(self, avg_reward, avg_q):
        self.eval_reward_last = avg_reward
        self.eval_avg_q_last = avg_q

        if avg_reward > self.eval_reward_record:
            self.eval_reward_record = avg_reward
            self.eval_avg_q_record = avg_q

    def reset_statistics(self):
        self.record_episode = 0
        self.record_episode_time = 0
        self.record_kill_count = 0

        self.reward_last_10 = collections.deque(list(), maxlen=10)
        self.reward_last_100 = collections.deque(list(), maxlen=100)
        self.reward_last_1000 = collections.deque(list(), maxlen=1000)
        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

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
