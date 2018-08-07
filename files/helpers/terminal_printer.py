from serpent.utilities import clear_terminal
import numpy as np
import time


class TerminalPrinter:
    """
    TerminalPrinter contain methods to print status screens for relevant contexts during play

    Printer class is set up to print entire screens of text at once to limit stuttering in console output
    """

    def __init__(self, replay_memory):
        self.lines = list()
        self.game_state = None
        self.replay_memory = replay_memory

    def add(self, content):
        self.lines.append(content)

    def empty_line(self):
        self.lines.append("")

    def clear(self):
        self.lines = list()

    def flush(self):
        clear_terminal()
        print("\n".join(self.lines))
        self.clear()

    def add_printer_head(self):
        run_time = time.time() - self.game_state.started_at
        # serpent.utilities.clear_terminal()

        #self.add("")
        self.add("Capgemini Intelligent Automation - Death Finger One Punch agent")
        self.add("Reinforcement Learning: Training a DQN Agent")
        self.add("")

        self.add("\033c" + f"SESSION RUN TIME: "
                 f"{round(run_time // 86400)} days, "
                 f"{round(run_time // 3600)} hours, "
                 f"{round((run_time // 60) % 60)} minutes, "
                 f"{round(run_time % 60)} seconds")
        self.add("")
        self.add(f"Current episode: {self.replay_memory.transitions.cur_episode}")
        self.add("")

        self.add(f"Average Last 10   Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_10, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_10, 2))}")
        self.add(f"Average Last 100  Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_100, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_100, 2))}")
        self.add(f"Average Last 1000 Runs: "
                         f" Time: {round(self.replay_memory.episode_statistics.average_time_1000, 2)}"
                         f" & Kill Count: {int(round(self.replay_memory.episode_statistics.average_kill_count_1000, 2))}")
        self.add("")

    def print_statistics(self, action, reward, move_time, agent_training, global_state_count, action_meaning_ofdp,
                         last_updated_net_time, last_updated_net_episode):
        episode_run_time_seconds = time.time() - self.game_state.episode_start_time
        #effective_apm = 0 if state_count else round(
        #    state_count / episode_run_time_seconds, 2)
        effective_apm = self.replay_memory.t / episode_run_time_seconds
        self.add_printer_head()

        #self.add(f"Reading context: {self.context}")

        self.add(f"AGENT DEATH FINGER ONE PUNCH THINKS:")
        self.add(f"Decisions made this episode: {self.replay_memory.t} "
                         f"\nAction: {action_meaning_ofdp[action]}")
                         #f"\nQ_values: {np.round(q_values, 2)}"  # {np.round(self.replay_memory.episode_memory.q_values[-1], 10)}"
                         #f"\nQ_values: [NOOP, LEFT, RIGHT]"  # , 2xLEFT, 2xRIGHT, R+L, L+R]"
                         #f"\nQ_value is how good I think each move is right now")

        #current_reward = self.calculate_reward()
        reward_feedback = ''
        if reward > 0:
            reward_feedback = 'Rewarded'
        elif reward < 0:
            reward_feedback = 'punished'

        self.add("")
        agent_mode = agent_training  # self.game_state.agent_mode  # "training" if self.game_state.training is True else "testing"
        self.add(f"Current task: {self.game_state.agent_mode}")
        self.add(f"So far observed: {global_state_count} states") #self.replay_memory.transitions.global_state_count} states")
        self.add(f"Agent is running in {agent_mode} mode")
        #self.add(f"Agent makes a random move {int(round(self.game_state.epsilon*100,0))}% of the time")
        self.add(f"Kill count   : {self.game_state.kill_count}")
        #self.add(f"Miss count   : {self.miss_count}")
        self.add(f"Player health: {self.game_state.health}")
        #self.add(f"Reward       : {reward}")
        self.add(f"Agent will be: {reward_feedback}")
        self.add(f"Total Reward : {self.game_state.episode_reward_total}")
        #self.add(f"Last 10 episode rewards: {list(self.replay_memory.episode_statistics.reward_last_10)}")
        #self.add(f"Loss from last training update: {training_loss}")
        self.add("")

        # TEMP PRINT FOR DEBUGGING
        self.add(f"Computing move in: {round(move_time, 3)} seconds")
        self.add(f"Effective decisions per second: {round(effective_apm, 2)}")
        # self.add(f"Episode clock time: "
        #                 f"{self.episode_time // 3600} hours, "
        #                 f"{(self.episode_time // 60) % 60} minutes, "
        #                 f"{self.episode_time % 60} seconds")
        #self.add(f"States processed this episode: {len(self.replay_memory.episode_memory.states)}")

        #if self.replay_memory.num_used > 0:
        self.add("")
        self.add(f"CURRENT RUN - TIME ALIVE: "
                 f"{round(episode_run_time_seconds, 2)} seconds "
                 f"and {self.game_state.kill_count} kills")
        self.add(f"LAST    RUN - TIME ALIVE: "
                 f"{round(self.replay_memory.episode_statistics.last_episode_time, 2)} seconds "
                 f"and {self.replay_memory.episode_statistics.last_episode_kill_count} kills")
        self.add(f"RECORD  RUN - TIME ALIVE: "
                 f"{round(self.replay_memory.episode_statistics.record_episode_time, 2)} seconds "
                 f"and {self.replay_memory.episode_statistics.record_kill_count} kills "
                 f"(Run {self.replay_memory.episode_statistics.record_episode})")
        self.add(f"LAST   EVALUATION - average reward: "
                 f"{round(self.replay_memory.episode_statistics.eval_reward_last, 2)} "
                 f"and average Q value {round(self.replay_memory.episode_statistics.eval_avg_q_last, 2)}")
        self.add(f"RECORD EVALUATION - average reward: "
                 f"{round(self.replay_memory.episode_statistics.eval_reward_record, 2)} "
                 f"and average Q value {round(self.replay_memory.episode_statistics.eval_avg_q_record, 2)}")
        self.add(f"last swapped online net : {last_updated_net_time} after {last_updated_net_episode} episodes")

        # Finally print all above to the screen as one message (Less flickering)
        self.flush()

    def print_error(self):
        self.add_printer_head()
        self.add("Memory address changed for this run")
        self.add("Agent will not be playing until kill count is correct again")
        self.add("This should be resolved within one or two runs")
        self.flush()
