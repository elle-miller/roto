import matplotlib.pyplot as plt
import numpy as np
import os

from tbparse import SummaryReader

plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('bmh')




from matplotlib import rcParams

# Set up LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 24
})


plt.style.use('bmh')


plt.rcParams.update(plt.rcParamsDefault)

# Set DPI
dpi = 600

plot_raw = False



# suprim
log_dir = "/home/emil/code/external/IsaacLab/multimodal_gym/paper_results"
output_dir = "/home/emil/code/external/IsaacLab/multimodal_gym/results/plots"

# agenst5
log_dir = "/home/emil/code/external/multimodal_gym/paper_results"
output_dir = "/home/emil/code/external/multimodal_gym/results"

# local
log_dir = "/home/elle/code/external/roto/results"
output_dir = "/home/elle/code/external/roto/results"

import plot_dicts
from plot import get_readers_seeded, get_min_length

### PAPER PLOTS ###
group_dict = plot_dicts.study_plot
group_dict = plot_dicts.tactile_weight
# group_dict = plot_dicts.franka_extra_plot
# group_dict = plot_dicts.memory_plot

exp_list = group_dict["exp_list"]

assert len(exp_list) == 3

# Create figure and subplots (1 row, 3 columns)
horizontal=True
if horizontal:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
else:
    fig, axes = plt.subplots(3, 1, figsize=(5, 6))


def get_readers(parent_experiment_dir, experiment_name):

    readers = []
    experiment_dir = os.path.join(parent_experiment_dir, experiment_name)
    
    num_seeds = len(sorted(os.listdir(experiment_dir)))

    for i in range(num_seeds):
        event_file = sorted(os.listdir(experiment_dir))[i]
        event_file = os.path.join(experiment_dir, event_file)
        print(event_file)
        reader = SummaryReader(event_file)  # long format
        readers.append(reader)
    return readers


# loop through franka, bounce, baoding
for ax_idx, exp in enumerate(exp_list):
    exp_dict = exp
    experiment_names = exp_dict["run_names"]

    for i, experiment_name in enumerate(experiment_names):

        parent_experiment_dir = os.path.join(log_dir, exp_dict["log_dir"], exp_dict["experiment_name"])


        print(i, experiment_name)

        readers = get_readers(parent_experiment_dir, experiment_name)

        tag = exp_dict["tag"]
        min_length = get_min_length(readers, tag)   

        # min_length=100
        mean_rewards = []
        step = []
        for reader in readers:
            df = reader.scalars
            metric = df[df["tag"] == exp_dict["tag"]]
            
            step = metric["step"].to_numpy() / 1e6
            metric = metric["value"].to_numpy()
            
            if tag == "num_rotations":
                run_rewards = metric /2
            elif tag == "timesteps_to_find_object_easy" or tag == "timesteps_to_find_object_med" or tag == "timesteps_to_find_object_hard":
                # 1 timestep = 1/60 seconds
                run_rewards = metric / 60
            else:
                run_rewards = metric
            mean_rewards.append(run_rewards)
            # print("Num rewards", np.shape(run_rewards))
            # axes[ax_idx].plot(step, run_rewards, linewidth=1.5, label=exp_dict["legend_names"][i], color=exp_dict["color"][i], linestyle=exp_dict["linestyle"][i])
            # Plot the mean and standard deviation
            if plot_raw:
                # Alternative: Use both sparse markers and thin connecting lines
                sample_rate = 1
                axes[ax_idx].plot(
                    step[::sample_rate], 
                    run_rewards[::sample_rate],
                    # marker=exp_dict["marker"][i],
                    # markersize=4,
                    # linewidth=0.1,
                    linestyle=exp_dict["linestyle"][i],
                    color=exp_dict["color"][i],
                )
        # Calculate the mean and standard deviation across the seeds
        mean = np.mean(mean_rewards, axis=0)
        std_dev = np.std(mean_rewards, axis=0)
        axes[ax_idx].plot(step, mean, label=exp_dict["legend_names"][i], color=exp_dict["color"][i], linestyle=exp_dict["linestyle"][i], )
        print(exp_dict["legend_names"][i])
        axes[ax_idx].fill_between(step, mean - std_dev, mean + std_dev, alpha=group_dict["opacity"], color=exp_dict["color"][i])

        axes[ax_idx].set_title(exp_dict["title"])
        y_max = exp_dict["y_max"]
        y_min = exp_dict["y_min"]
        x_max = exp_dict["x_max"]  
        xmin = exp_dict["x_min"]  

        print("xlim", xmin)
        axes[ax_idx].set_xlim(xmin,x_max) 
        axes[ax_idx].set_ylim(y_min, y_max)   
        axes[ax_idx].grid(True, alpha=0.3)  # Reduced grid opacity to 30%

        # axes[ax_idx].set_ylabel(exp_dict["y_label"])

# # if max reward, plot this
# if exp_dict["max_reward"] > 0:
#     # Add a horizontal line: dashed style, reduced opacity
#     # plt.axhline(y=exp_dict["max_reward"], color='red', linestyle='--', alpha=0.5, label="Max return")  
#     plt.axhline(y=exp_dict["max_reward"], color='red', linestyle='--', alpha=0.5)  

# Optional: add a small margin (0.05 or 5% here) to avoid cutting off plot lines

# plt.ylim([0, 350])
# plt.xlim([0,5e6])
axes[1].set_xlabel("Timesteps (M)")
axes[0].set_ylabel(group_dict["y_label"])
# axes[0].legend(loc="lower right", framealpha=1.0)
# axes[0].legend(loc="upper right", framealpha=1.0)

axes[group_dict["legend_idx"]].legend(loc=group_dict["legend_loc"], framealpha=1.0)
# axes[0].legend(loc="upper right")


# axes[1].legend(loc=group_dict["legend_loc"])

# # plt.ylim(y_min, y_max)
# plt.autoscale(tight=True)
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)  # Reduce horizontal spacing between subplots

plt.savefig(os.path.join(output_dir, group_dict["output_file"]), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
plt.show()


