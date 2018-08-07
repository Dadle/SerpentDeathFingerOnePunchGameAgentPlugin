import os
import plotly
from plotly.graph_objs import Scatter, Line
import torch


# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10


# Test DQN
def test(args, T_period, dqn, val_mem, evaluate=False):
    global Ts, rewards, Qs, best_avg_reward
    Ts.extend(T_period)
    T_Qs = []
    T_rewards = list(val_mem.episode_statistics.reward_last_10)
    print("stats last 10 rewards:", val_mem.episode_statistics.reward_last_10)

    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        T_Qs.append(dqn.evaluate_q(state))

    avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
    if not evaluate:
        # Append to results
        rewards.extend(T_rewards)
        Qs.extend(T_Qs)

        # Plot
        print("plotting rewards:", rewards)
        print("against Ts:", Ts)
        _plot_line(Ts, rewards, 'Reward', path='logs')
        print("plotting Qs:", Qs)
        _plot_line(Ts, Qs, 'Q', path='logs')

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            dqn.save(os.path.join(args.checkpoint_path, args.env_name))

    # Return average reward and Q-value
    return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  print("ys", ys)
  print('stats: min', ys.min()[0], 'max', ys.max()[0], 'mean', ys.mean(), 'std', ys.std())
  #ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_min, ys_max, ys_mean, ys_std = ys.min()[0], ys.max()[0], ys.mean(), ys.std()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='lightcyan'), name='+1 Std. Dev.', showlegend=True)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='lightgreen'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
