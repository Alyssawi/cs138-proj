import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_rewards_from_dir(directory, tag="eval/mean_reward"):
    reward_curves = []
    steps_curves = []

    event_files = glob.glob(os.path.join(directory, "**", "events.*"), recursive=True)

    for file in event_files:
        ea = EventAccumulator(file)
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            rewards = np.array([e.value for e in events])
            reward_curves.append(rewards)
            steps_curves.append(steps)

    return steps_curves, reward_curves


def align_and_average(steps_list, rewards_list):
    common_steps = np.linspace(0, max(max(s) for s in steps_list), num=500)
    interpolated_rewards = []

    for steps, rewards in zip(steps_list, rewards_list):
        interp_rewards = np.interp(common_steps, steps, rewards)
        interpolated_rewards.append(interp_rewards)

    mean_rewards = np.mean(interpolated_rewards, axis=0)
    return common_steps, mean_rewards


def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    base_dir = "base/"
    curr_dir = "curr-softmax-noise-2/"

    base_steps, base_rewards = extract_rewards_from_dir(base_dir)
    curr_steps, curr_rewards = extract_rewards_from_dir(curr_dir)

    base_x, base_y = align_and_average(base_steps, base_rewards)
    curr_x, curr_y = align_and_average(curr_steps, curr_rewards)

    smoothing_weight = 0.75
    base_y_smooth = smooth_curve(base_y, weight=smoothing_weight)
    curr_y_smooth = smooth_curve(curr_y, weight=smoothing_weight)

    step_min = 0
    step_max = 100000
    base_mask = (base_x >= step_min) & (base_x <= step_max)
    curr_mask = (curr_x >= step_min) & (curr_x <= step_max)

    plt.figure(figsize=(10, 6))
    plt.plot(
        base_x[base_mask],
        base_y_smooth[base_mask],
        label="Base",
        linewidth=2,
    )
    plt.plot(
        curr_x[curr_mask],
        curr_y_smooth[curr_mask],
        label="Softmax Noise",
        linewidth=2,
    )
    plt.xlabel("Steps")
    plt.ylabel("Mean Reward")
    plt.title("eval/mean_reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
