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

    interpolated_rewards = np.array(interpolated_rewards)
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    return common_steps, mean_rewards, std_rewards


def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    base_dir = "base/"
    curr_dir = "curr-softmax-noise-kappa-0.50/"

    base_steps, base_rewards = extract_rewards_from_dir(base_dir)
    curr_steps, curr_rewards = extract_rewards_from_dir(curr_dir)

    base_x, base_y, base_std = align_and_average(base_steps, base_rewards)
    curr_x, curr_y, curr_std = align_and_average(curr_steps, curr_rewards)

    smoothing_weight = 0.75
    base_y_smooth = smooth_curve(base_y, weight=smoothing_weight)
    curr_y_smooth = smooth_curve(curr_y, weight=smoothing_weight)
    base_std_smooth = smooth_curve(base_std, weight=smoothing_weight)
    curr_std_smooth = smooth_curve(curr_std, weight=smoothing_weight)

    step_min = 0
    step_max = 250000
    base_mask = (base_x >= step_min) & (base_x <= step_max)
    curr_mask = (curr_x >= step_min) & (curr_x <= step_max)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        base_x[base_mask],
        base_y_smooth[base_mask],
        label="Baseline",
        color="#1f77b4",
        linewidth=3,
    )
    # ax.fill_between(
    #     base_x[base_mask],
    #     base_y_smooth[base_mask] - base_std_smooth[base_mask],
    #     base_y_smooth[base_mask] + base_std_smooth[base_mask],
    #     color="#1f77b4",
    #     alpha=0.2,
    # )
    ax.plot(
        curr_x[curr_mask],
        curr_y_smooth[curr_mask],
        label="EGT ($\\kappa = 0.5$)",
        color="#ff7f0e",
        linewidth=3,
    )
    # ax.fill_between(
    #     curr_x[curr_mask],
    #     curr_y_smooth[curr_mask] - curr_std_smooth[curr_mask],
    #     curr_y_smooth[curr_mask] + curr_std_smooth[curr_mask],
    #     color="#ff7f0e",
    #     alpha=0.2,
    # )

    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("Mean Reward", fontsize=14)
    ax.set_title("\\textbf{Evaluation Reward Over Time}", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    # Adjust subplot margins and save without padding
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.12)
    fig.savefig("reward_plot.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
