import matplotlib.pyplot as plt
import numpy as np
import os

from tbparse import SummaryReader

plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use('bmh')
plt.style.use('ggplot')  # Good for accessibility
# Set up LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 24
})


plt.rcParams.update(plt.rcParamsDefault)

fig, ax = plt.subplots(figsize=(5, 4))
# fig, ax = plt.subplots(figsize=(6, 4))


from matplotlib import rcParams




# Set DPI
dpi = 600

plot_raw = True

# suprim
log_dir = "/home/emil/code/external/IsaacLab/multimodal_gym/paper_results"
output_dir = "/home/emil/code/external/IsaacLab/multimodal_gym/results/plots"

# agenst5
# log_dir = "/home/emil/code/external/multimodal_gym/paper_results"
# output_dir = "/home/emil/code/external/multimodal_gym/results"

import plot_dicts



def get_readers_seeded(parent_experiment_dir, experiment_name):

    readers = []
    experiment_dir = os.path.join(parent_experiment_dir, experiment_name)
    
    # loop through all individual runs, getting min length
    for seed in os.listdir(experiment_dir):
        seed_dir = os.path.join(experiment_dir, seed)
        # ['checkpoints', 'events.out.tfevents.1737560832.suprim.859756.0', 'params']
        event_file = sorted(os.listdir(seed_dir))[1]
        event_file = os.path.join(seed_dir, event_file)
        print(event_file)
        reader = SummaryReader(event_file)  # long format
        readers.append(reader)
    return readers

def get_readers(parent_experiment_dir, experiment_name):

    readers = []
    experiment_dir = os.path.join(parent_experiment_dir, experiment_name)
    
    num_seeds = len(sorted(os.listdir(experiment_dir)))

    for i in range(num_seeds):
        event_file = sorted(os.listdir(experiment_dir))[i]
        event_file = os.path.join(experiment_dir, event_file)
        print(event_file)
        reader = SummaryReader(event_file)  # long format
        print("Read:", event_file)
        readers.append(reader)
    return readers


def get_min_length(readers, tag):
    min_length = 10000
    for reader in readers:
        df = reader.scalars
        # print(df)
        rewards = df[df["tag"] == tag]["value"].to_numpy()
        min_length = np.min((min_length, np.shape(rewards)[0]))
        print(min_length)
    print("min length", min_length)
    return min_length



def main():
    exp_dict = plot_dicts.baoding_ssl_rotations_barchart

    exp_dict = plot_dicts.tp_rate

    experiment_names = exp_dict["run_names"]

    max_values = []
    mean_values = []

    for i, experiment_name in enumerate(experiment_names):

        parent_experiment_dir = os.path.join(log_dir, exp_dict["log_dir"], exp_dict["experiment_name"])


        print(i, experiment_name)

        readers = get_readers(parent_experiment_dir, experiment_name)

        mean_rewards = []
        step = []
        for reader in readers:
            df = reader.scalars

            if "tags" in exp_dict.keys():
                tags = exp_dict["tags"]
            else:
                tags = [exp_dict["tag"]]

            for j, tag in enumerate(tags):

                # print(j, tag)
                # if not tag in df.tag:
                #     print(tag, "not in tags")
                #     continue

                try:
                
                    metric = df[df["tag"] == tag]
                except:
                    continue
            
                step = metric["step"].to_numpy() # / 1e6
                metric = metric["value"].to_numpy()
                
                if tag == "num_rotations":
                    run_rewards = metric /2
                elif tag == "timesteps_to_find_object_easy" or tag == "timesteps_to_find_object_med" or tag == "timesteps_to_find_object_hard":
                    # 1 timestep = 1/60 seconds
                    run_rewards = metric / 60
                elif tag == "bounce_reward":
                    run_rewards = metric / 10
                else:
                    run_rewards = metric
                
                mean_rewards.append(run_rewards)
                print("Num values", np.shape(run_rewards))
                if plot_raw:
                    label = None if j > 0 else exp_dict["legend_names"][i]
                    plt.plot(step, run_rewards, alpha=1-j*0.1, color=exp_dict["color"][i], label=label, linestyle=exp_dict["linestyle"][i])

        
        if "bar_chart" in exp_dict.keys():
            mean_rewards = np.array(mean_rewards)
            max_value = np.max(mean_rewards)
            print(np.shape(mean_rewards))

            if tag == "timesteps_to_find_object_easy" or tag == "timesteps_to_find_object_med" or tag == "timesteps_to_find_object_hard":
                
                # weird timestep in beginning
                mean_rewards = mean_rewards[:, 5:]

                # subtract to get time remaining
                mean_rewards = mean_rewards

                mean = np.mean(mean_rewards, axis=0)
                min_mean = np.min(mean)
                min_value = np.min(mean_rewards)

                min_mean = 5- min_mean
                min_value = 5-min_value

                print("min_mean", min_mean)
                print("min", min_value)

                mean_values.append(min_mean)
                max_values.append(min_value)
            else:
                mean = np.mean(mean_rewards, axis=0)
                max_mean = np.max(mean)
                mean_values.append(max_mean)
                max_values.append(max_value)
            continue

        # Plot the mean and standard deviation
        if not plot_raw:
            # Calculate the mean and standard deviation across the seeds
            mean = np.mean(mean_rewards, axis=0)
            std_dev = np.std(mean_rewards, axis=0)
            plt.plot(step, mean, label=exp_dict["legend_names"][i], color=exp_dict["color"][i], linestyle=exp_dict["linestyle"][i])
            print(exp_dict["legend_names"][i])
            plt.fill_between(step, mean - std_dev, mean + std_dev, alpha=0.08, color=exp_dict["color"][i])

    

    if "bar_chart" in exp_dict.keys():
        # Create bar chart
        bars = ax.bar(exp_dict["legend_names"], max_values, color=exp_dict["color"], width=0.6, alpha=0.25)
        bars2 = ax.bar(exp_dict["legend_names"], mean_values, width=0.6, color=exp_dict["color"], alpha=0.75)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a light background color to highlight the chart area
        ax.set_facecolor('#f8f9fa')
        ax.tick_params(axis='x', labelsize=8)  # You can adjust the size (8) as needed
        # ax.legend(loc=exp_dict["legend_loc"])

    else:
        x_max = exp_dict["x_max"]
        x_min = exp_dict["x_min"]
        # plt.xlabel("Timestep (M)")
        plt.xlabel(exp_dict["x_label"])
        plt.xlim(x_min, x_max)


    # Optional: add a small margin (0.05 or 5% here) to avoid cutting off plot lines
    y_max = exp_dict["y_max"]
    y_min = exp_dict["y_min"]


    # plt.tight_layout()
    # plt.ylim([0, 350])
    # plt.xlim([0,5e6])
    # plt.ylabel(exp_dict["y_label"])
    plt.legend(loc=exp_dict["legend_loc"])
    plt.title(exp_dict["title"])
    plt.grid(True, alpha=0.3)  # Reduced grid opacity to 30%
    ax.set_facecolor('white')

    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, exp_dict["output_file"]), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def get_metrics(exp_dict):
    max_values = []
    mean_values = []

    experiment_names = exp_dict["run_names"]


    for i, experiment_name in enumerate(experiment_names):

        parent_experiment_dir = os.path.join(log_dir, exp_dict["log_dir"], exp_dict["experiment_name"])


        print(i, experiment_name)

        readers = get_readers(parent_experiment_dir, experiment_name)

        # min_length=100
        mean_rewards = []
        step = []
        for reader in readers:
            df = reader.scalars

            # tag = exp_dict["tag"]
            tags = exp_dict["tags"]
            for tag in enumerate(tags):

                metric = df[df["tag"] == tag]
                
                step = metric["step"].to_numpy() / 1e6
                metric = metric["value"].to_numpy()
                
                if tag == "num_rotations":
                    run_rewards = metric /2
                elif tag == "timesteps_to_find_object_easy" or tag == "timesteps_to_find_object_med" or tag == "timesteps_to_find_object_hard":
                    # 1 timestep = 1/60 seconds
                    run_rewards = metric / 60
                elif tag == "bounce_reward":
                    run_rewards = metric / 10
                else:
                    run_rewards = metric
                mean_rewards.append(run_rewards)
                # print("Num rewards", np.shape(run_rewards))
                if plot_raw:
                    plt.plot(step, run_rewards, label=exp_dict["legend_names"][i])

        
        if "bar_chart" in exp_dict.keys():
            mean_rewards = np.array(mean_rewards)
            max_value = np.max(mean_rewards)
            print(np.shape(mean_rewards))

            if tag == "timesteps_to_find_object_easy" or tag == "timesteps_to_find_object_med" or tag == "timesteps_to_find_object_hard":
                
                # weird timestep in beginning
                mean_rewards = mean_rewards[:, 5:]

                # subtract to get time remaining
                mean_rewards = mean_rewards

                mean = np.mean(mean_rewards, axis=0)
                min_mean = np.min(mean)
                min_value = np.min(mean_rewards)

                min_mean = 5- min_mean
                min_value = 5-min_value

                print("min_mean", min_mean)
                print("min", min_value)

                mean_values.append(min_mean)
                max_values.append(min_value)
            else:
                mean = np.mean(mean_rewards, axis=0)
                max_mean = np.max(mean)
                mean_values.append(max_mean)
                max_values.append(max_value)
            continue

    return mean_values, max_values


if __name__ == "__main__":

    main()