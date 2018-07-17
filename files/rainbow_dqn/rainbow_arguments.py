class RainbowArguments:
    """
    Contains all artuments used by RainbowDQN
    To change these arguments either change the defaults in this file or provide keyword arguments upon initialization
    """

    def __init__(self, **kwargs):
        self.seed = 123  # Random seed
        self.disable_cuda = False # Set true to disable CUDA
        self.T_max = int(50e6) # Number of training steps (4x number of frames)
        self.max_episode_length = int(108e3) # Max episode length (0 to disable)
        self.history_length = 4 # Number of consecutive states processed
        self.hidden_size = 512 # Network hidden size
        self.noisy_std = 0.1 # Initial standard deviation of noisy linear layers
        self.atoms = 51 # Discretised size of value distribution
        self.V_min = -10 # Minimum of value distribution support
        self.V_max = 10 # Maximum of value distribution support')
        self.model = None # Pretrained model (state dict) TODO: Verify use of this parameter
        self.memory_capacity = int(1e6) # Experience replay memory capacity
        self.replay_frequency = 4 # Frequency of sampling from memory
        self.priority_exponent = 0.5 # Prioritised experience replay exponent (originally denoted α)
        self.priority_weight = 0.4 # Initial prioritised experience replay importance sampling weight
        self.multi_step = 3 #Number of steps for multi-step return
        self.discount = 0.99 # Discount factor
        self.target_update = int(32e3) # Number of steps after which to update target network
        self.reward_clip = 1 # Reward clipping (0 to disable)
        self.lr = 0.0000625 # Learning rate
        self.adam_eps = 1.5e-4 # Adam epsilon
        self.batch_size = 32 # Batch size
        self.norm_clip = 10 # Max L2 norm for gradient clipping
        self.learn_start = int(80e3) # Number of steps before starting training
        self.evaluate = False # True means evaluate only
        self.evaluation_interval = 100000 # Number of training steps between evaluations
        self.evaluation_episodes = 10 # Number of evaluation episodes to average over
        self.evaluation_size = 500 # Number of transitions to use for validating Q
        self.log_interval = 25000 # Number of training steps between logging status

        for key, value in kwargs.items():
            setattr(self, key, value)
