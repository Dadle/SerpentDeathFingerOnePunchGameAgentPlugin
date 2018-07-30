
from datetime import datetime
import random
import torch

from .rainbow_agent import Agent
from .replay_memory import ReplayMemory
from .rainbow_arguments import RainbowArguments
from .test import test


class RainbowMain:

    def __init__(self, **kwargs):
        self.args = RainbowArguments(kwargs)

        self.random.seed(self.args.seed)
        torch.manual_seed(random.randint(1, 10000))

        # Configure Torch backend
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.args.device = torch.device('cuda')
            torch.cuda.manual_seed(random.randint(1, 10000))
            torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
        else:
            self.args.device = torch.device('cpu')


    # Simple ISO 8601 timestamped logger
    @staticmethod
    def log(s):
        print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

    def setup_agent(self, number_of_possible_actions):
        # Environment
        env = Env(args)
        env.train()
        action_space = env.action_space()

        # Agent
        dqn = Agent(self.args, number_of_possible_actions) # TODO: Important
        mem = ReplayMemory(self.args, self.args.memory_capacity)
        priority_weight_increase = (1 - self.args.priority_weight) / (self.args.T_max - self.args.learn_start)

        # Construct validation memory
        val_mem = ReplayMemory(self.args, self.args.evaluation_size)
        T, done = 0, True
        # TODO: This is the gameplay loop if evaluation is true
        while T < self.args.evaluation_size:
            if done:
                state, done = env.reset(), False

            next_state, _, done = env.step(random.randint(0, action_space - 1))
            val_mem.append(state, None, None, done)
            state = next_state
            T += 1

        if args.evaluate:
            dqn.eval()  # Set DQN (online network) to evaluation mode
            avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)  # Test
            print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        else:
            # Training loop
            # TODO: This is the gameplay loop for a training agent
            dqn.train()
            T, done = 0, True
            while T < self.args.T_max:
                if done:
                    state, done = env.reset(), False

                if T % self.args.replay_frequency == 0:
                    dqn.reset_noise()  # Draw a new set of noisy weights

                action = dqn.act(state)  # Choose an action greedily (with noisy weights)
                next_state, reward, done = env.step(action)  # Step
                if self.args.reward_clip > 0:
                    reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
                mem.append(state, action, reward, done)  # Append transition to memory
                T += 1

                if T % self.args.log_interval == 0:
                    self.log('T = ' + str(T) + ' / ' + str(self.args.T_max))

                # Train and test
                if T >= self.args.learn_start:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase,
                                              1)  # Anneal importance sampling weight β to 1

                    if T % self.args.replay_frequency == 0:
                        dqn.learn(mem)  # Train with n-step distributional double-Q learning

                    if T % self.args.evaluation_interval == 0:
                        dqn.eval()  # Set DQN (online network) to evaluation mode
                        avg_reward, avg_Q = test(self.args, T, dqn, val_mem)  # Test
                        self.log('T = ' + str(T) + ' / ' + str(self.args.T_max) + ' | Avg. reward: ' + str(
                            avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                        dqn.train()  # Set DQN (online network) back to training mode

                    # Update target network
                    if T % self.args.target_update == 0:
                        dqn.update_target_net()

                state = next_state

        env.close()
