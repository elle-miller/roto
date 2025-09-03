 


franka_find = {
    "title": "Franka Find",
    "log_dir": "franka_find",
    "legend_names": ["PPO"],# , "continuous"],
    "experiment_names": ["franka_find_e2e_binary_seeded"],
    "tag": "mean_eval_return",
    "output_file": "franka_find_e2e.png",
    "color": ["tab:blue", "tab:green", "tab:red"],
    "linestyle": ['-', '-', '-'],
    "max_reward": 275,
    "y_max": 300  

}

bounce_baselines = {
    "title": "Shadow Bounce",
    "log_dir": "shadow",
    "legend_names": ["PPO"], # "continuous"],
    "experiment_names": ["shadow_bounce_e2e_binary_seeded"], #"shadow_bounce_e2e_cont_seeded"],
    "tag": "mean_eval_return",
    "output_file": "shadow_bounce_e2e.png",
    "color": ["tab:blue", "tab:green", "tab:red"],
    "linestyle": ['-', '-', '-'],
    "max_reward": 600,
    "y_max": 640
}

baoding_baselines = {
    "title": "Shadow Baoding",
    "log_dir": "shadow",
    "legend_names": ["PPO"], # "continuous"],
    "experiment_names": ["shadow_baoding_binary_e2e_seeded2"], #"shadow_bounce_e2e_cont_seeded"],
    "tag": "mean_eval_return",
    "output_file": "shadow_baoding_e2e.png",
    "color": ["tab:blue", "tab:green", "tab:red"],
    "linestyle": ['-', '-', '-'],
    "max_reward": 0,
    "y_max": 1200
}

franka_reward = {
    "title": "Franka Find",
    "log_dir": "franka_find",
    "legend_names": ["prop"],# , "continuous"],
    "experiment_names": ["franka_find_e2e_prop"],
    "tag": "object_found_med",
    "output_file": "franka_find_object_found_med.png",
    "color": ["tab:blue", "tab:red"],
    "linestyle": ['-', '-', '-'],
    "max_reward": 275,
    "y_max": 1,  
    "y_min": 0,
    "y_label": "object found %"  

}


colors = ['forestgreen','crimson', "royalblue"]
colors = ['#6a5acd', '#20b2aa', '#ff7f50']  # Slate Blue, Light Sea Green, Coral
colors = ['#8e44ad', '#16a085', '#d35400']  # Purple, Teal, Orange
colors = ['#64b5f6', '#81c784', '#ba68c8']  # Sky Blue, Mint Green, Lavender Purple
colors = ['#9575cd', '#4db6ac', '#f06292']  # Medium Purple, Teal, Pink
colors = ['#80deea', '#9fa8da', '#ffcc80']  # Light Cyan, Periwinkle, Light Orange
colors = ['#64b5f6', '#9575cd', '#f06292']  # sky blue,Medium Purple, Pink

ppo_colour=colors[0]
recon_colour="red" #colors[1]
dynamics_colour="tab:green" #colors[2]
memory_colour="purple"

franka_stack = {
    "title": "Franka Find",
    "log_dir": "franka_find",
    "legend_names": ["s1", "s4", "s8", "s32"],# , "continuous"],
    "experiment_names": ["franka_find_e2e_pt_s1", "franka_find_e2e_pt_s4", "franka_find_e2e_pt_s8", "franka_find_e2e_pt_s32"],
    "tag": "object_found_med",
    "output_file": "franka_find_object_found_med_stack.png",
    "color": ["tab:blue", "tab:red", "tab:green", "tab:green"],
    "linestyle": ['-', '-', '-', "-", "-"],
    "max_reward": 275,
    "y_max": 1,  
    "y_min": 0,
    "y_label": "object found %"  
}

franka_prop_stack = {
    "title": "Find",
    "log_dir": "part_1",
    "experiment_name": "franka_find",
    "legend_names": ["p_s16", "pt_s16",  "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1",  "p_s16_no_action"],
    "run_names": ["p_s16", "pt_s16",  "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1", "p_s16_no_action"],
    "tag": "mean_eval_return", # object_found_easy
    "output_file": "franka_find_part_1.png",
    "color": ["tab:blue", "tab:blue", "tab:red", "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['--', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "y_max": 225,  
    "y_min": 50,
    "y_label": "%",
    "legend_loc": "upper left"
}

bounce_prop_stack = {
    "title": "Bounce",
    "log_dir": "part_1",
    "experiment_name": "shadow_bounce",
    "legend_names": ["p_s16", "pt_s16",  "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1"],
    "run_names": ["p_s16", "pt_s16", "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1"],
    "tag": "mean_eval_return", # object_found_easy
    "output_file": "part_1.png",
    "color": ["tab:blue", "tab:blue", "tab:red", "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['--', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "y_max": 700,  
    "y_min": -10,
    "y_label": "%",
    "legend_loc": "upper left"
}

baoding_prop_stack = {
    "title": "Baoding",
    "log_dir": "part_1",
    "experiment_name": "shadow_baoding",
    "legend_names": ["p_s16", "pt_s16",  "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1"],
    "run_names": ["p_s16", "pt_s16", "p_s4", "pt_s4", "p_s2", "pt_s2", "p_s1", "pt_s1"],
    "tag": "mean_eval_return", # object_found_easy
    "output_file": "part_1.png",
    "color": ["tab:blue", "tab:blue", "tab:red", "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['--', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "y_max": 420,  
    "y_min": 0,
    "y_label": "%",
    "legend_loc": "upper left"
}


franka_recon = {
    "title": "Franka Find",
    "log_dir": "franka_find",
    "legend_names": ["PPO", "+recon"],# , "continuous"],
    "experiment_names": ["franka_find_e2e_binary_seeded", "franka_find_recon_binary_seeded"],
    "tag": "mean_eval_return",
    "output_file": "franka_find_recon.png",
    "color": ["tab:blue", "tab:red"],
    "linestyle": ['-', '-', '-'],
    "max_reward": 275,
    "y_max": 200  
}

bounce_preds = {
    "title": "Bounce",
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["+tactile recon", "+tactile dynamics"],
    "run_names": ["tactile_recon", "tactile_dynamics"],
    "tag": "mean_eval_return", # object_found_easy
    "color": [recon_colour, dynamics_colour, dynamics_colour, "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '--', "--", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "y_max": 850,  
    "y_min": 0,
    "legend_loc": "upper left",
    "output_file": "bounce_preds.png",
    "y_label": "mean"
}

prop_colour = "orange"
prop_no_action_colour = "#DA70D6"

franka_study = {
    "title": "Find",
    "log_dir": "part_1",
    "experiment_name": "franka_find",
    "legend_names": ["prop-action-tactile", "prop-action", "prop"],
    "run_names": ["prop_tactile", "prop", "prop_no_action"],
    "tag": "object_found_med", #"mean_eval_return",
    "color": [ppo_colour, prop_colour, prop_no_action_colour],
    "linestyle": ['-', '-', ':'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 1,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "% object found within 1cm", #Mean evaluation return",

}

franka_study_mean_return = {
    "title": "Find",
    "log_dir": "part_1",
    "experiment_name": "franka_find",
    "legend_names": ["prop-tactile", "prop", "prop(no action)"],
    "run_names": ["prop_tactile", "prop", "prop_no_action"],
    "tag": "mean_eval_return", #"mean_eval_return",
    "color": [ppo_colour, prop_colour, prop_no_action_colour],
    "linestyle": ['-', '-', ':'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 215,  
    "y_min": 60,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return", #Mean evaluation return",

}

bounce_study = {
    "title": "Bounce",
    "log_dir": "part_1",
    "experiment_name": "shadow_bounce",
    "legend_names": ["prop-tactile", "prop",  "prop(no action)"],
    "run_names": ["prop_tactile", "prop_optimised", "prop_optimised_no_action"],
    "tag": "mean_eval_return",
    "color": [ppo_colour, prop_colour, prop_no_action_colour],
    "linestyle": ['-', '-', ":"],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 850,  
    "y_min": 0,
    "y_label": "Mean evaluation return",
    "legend_loc": "upper left"
}


baoding_study_mean_return = {
    "title": "Baoding",
    "log_dir": "part_1",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "PPO(prop)",  "PPO(prop(no last action))"],
    "run_names": ["e2e_3", "prop_optimised", "prop_no_action"],
    "tag": "mean_eval_return",
    "color": [ppo_colour, prop_colour, prop_no_action_colour],
    "linestyle": ['-', '-', ':'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 300,  
    "y_min": 0,
    "y_label": "Mean evaluation return",
    "legend_loc": "upper left",
    "output_file": "baoding_test.png"
}

prop_colour = "orange"


franka_ssl_timesteps = {
    "title": "within 1cm",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon",  "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_2", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics"],
    "tag": "timesteps_to_find_object_med", #, # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--"],
    # "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "x_min": 5,
    "x_max": 199,
    # "y_max": 230,  
    # "y_min": 90,
    "y_max": 5,  
    "y_min": 1,
    "legend_loc": "upper left",
    "y_label": "Seconds to find object", #Mean evaluation return",
}
franka_ssl_timesteps_easy = {
    "title": "within 3cm",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)",  "+full recon", "+tactile recon",  "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_2", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics"],
    "tag": "timesteps_to_find_object_easy", #, # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--"],
    # "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "x_min": 5,
    "x_max": 199,
    # "y_max": 230,  
    # "y_min": 90,
    "y_max": 5,  
    "y_min": 1,
    "legend_loc": "upper left",
    "y_label": "Seconds to find object", #Mean evaluation return",
}

franka_ssl_timesteps_hard = {
    "title": "within 0.05cm",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon",  "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_2", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics"],
    "tag": "timesteps_to_find_object_hard", #, # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--"],
    # "marker": ['.', 'o','.', 'o','.', 'o','.', 'o','.', 'o',],
    "max_reward": 0,
    "x_min": 5,
    "x_max": 199,
    # "y_max": 230,  
    # "y_min": 90,
    "y_max": 5,  
    "y_min": 1,
    "legend_loc": "upper left",
    "y_label": "Seconds to find object", #Mean evaluation return",
}


franka_ssl_mean_return = {
    "title": "Find",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon",  "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_2", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics"],
    "tag": "mean_eval_return",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "purple", "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--", "-"],
    "max_reward": 0,
    "x_min": 3,
    "x_max": 199,
    "y_max": 235,  
    "y_min": 90,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

franka_memory_mean_return = {
    "title": "Find",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop-tactile)", "+full dynamics", "memory"],
    "run_names": ["e2e_2", "dynamics_2", "memory"],
    "tag": "mean_eval_return",
    "color": [ppo_colour, dynamics_colour, memory_colour, "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--", "-"],
    "max_reward": 0,
    "x_min": 3,
    "x_max": 199,
    "y_max": 235,  
    "y_min": 90,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

bounce_ssl_mean_return = {
    "title": "Bounce",
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO","PPO(proprioceptive)", "+recon", "+tactile recon", "+dynamics", "+tactile dynamics"],
    "run_names": ["e2e", "prop", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics_t2"], #fked up tactile dynamics trial
    "tag": "mean_eval_return", # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour,  dynamics_colour, "purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 900,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

bounce_sweep_example = {
    "title": "Bounce (+ full dynamics) sweep",
    "log_dir": "appendix",
    "experiment_name": "sweep_example",
    "legend_names": ["PPO"],
    "run_names": ["bounce_dynamics"],
    "tag": "mean_eval_return", # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour,  dynamics_colour, "purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 900,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
    "output_file": "example_sweep.pdf"
}

tp_rate = {
    "title": "TP rate",
    "log_dir": "appendix",
    "experiment_name": "tactile_dynamics",
    "legend_names": ["Find", "Bounce", "Baoding"],
    "run_names": ["franka_find", "shadow_bounce", "shadow_baoding"],
    "tags": ["Tactile / true_positive_rate @t=0", "Tactile / true_positive_rate @t=1", "Tactile / true_positive_rate @t=2", "Tactile / true_positive_rate @t=8"], # object_found_easy
    "color": [dynamics_colour, ppo_colour, recon_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 1,  
    "y_min": 0.88,
    "legend_loc": "upper right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "all_tacdyn_tp_rate.pdf"
}

bounce_fp_rate = {
    "title": "FP rate",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": [r"$t=1$", r"$t=2$", r"$t=3$", r"$t=9$"],
    "run_names": ["tactile_dynamics"],
    "tags": ["Tactile / false_positive_rate @t=0", "Tactile / false_positive_rate @t=1", "Tactile / false_positive_rate @t=2", "Tactile / false_positive_rate @t=8"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 0.3,  
    "y_min": 0.05,
    "legend_loc": "upper right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "bounce_tacdyn_fp_rate.pdf"
}

bounce_fn_rate = {
    "title": "FN rate",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": [r"$t=1$", r"$t=2$", r"$t=3$", r"$t=9$"],
    "run_names": ["tactile_dynamics"],
    "tags": ["Tactile / false_negative_rate @t=0", "Tactile / false_negative_rate @t=1", "Tactile / false_negative_rate @t=2", "Tactile / false_negative_rate @t=8"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 0.005,  
    "y_min": 0.001,
    "legend_loc": "upper right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "bounce_tacdyn_fn_rate.pdf"
}

bounce_tn_rate = {
    "title": "TN rate",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": [r"$t=1$", r"$t=2$", r"$t=3$", r"$t=9$"],
    "run_names": ["tactile_dynamics"],
    "tags": ["Tactile / true_negative_rate @t=0", "Tactile / true_negative_rate @t=1", "Tactile / true_negative_rate @t=2", "Tactile / true_negative_rate @t=8"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 1,  
    "y_min": 0.96,
    "legend_loc": "lower right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "bounce_tacdyn_tn_rate.pdf"
}

bounce_accuracy = {
    "title": "Accuracy",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": [r"$t=1$", r"$t=2$", r"$t=3$", r"$t=9$"],
    "run_names": ["tactile_dynamics"],
    "tags": ["Tactile / accuracy @t=0", "Tactile / accuracy @t=1", "Tactile / accuracy @t=2", "Tactile / accuracy @t=8"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 1,  
    "y_min": 0.96,
    "legend_loc": "lower right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "bounce_tacdyn_accuracy.pdf"
}

bounce_f1 = {
    "title": "F1",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": [r"$t=1$", r"$t=2$", r"$t=3$", r"$t=9$"],
    "run_names": ["tactile_dynamics"],
    "tags": ["Tactile / f1 @t=0", "Tactile / f1 @t=1", "Tactile / f1 @t=2", "Tactile / f1 @t=8"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 700,
    "x_min": 0,
    "y_max": 0.96,  
    "y_min": 0.84,
    "legend_loc": "upper right",
    "x_label": "Number of rollouts",
    # "y_label": "Mean evaluation return",
    "output_file": "bounce_tacdyn_f1.pdf"
}

aux_loss_example = {
    "title": "Bounce preds",
    "log_dir": "appendix",
    "experiment_name": "shadow_bounce",
    "legend_names": ["0", "1", "2"],
    "run_names": ["tactile_dynamics"],
    "tag": "aux_loss", #tp_rate@t=0", "tp_rate@t=1", "tp_rate@t=2"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 150,
    "x_min": 60,
    "y_max": 1.9,  
    "y_min": 1.7,
    "legend_loc": "upper left",
    "y_label": "Average auxiliary loss",
    "output_file": "aux_loss.png"
}

baoding_dynamics_aux_loss = {
    "title": r"Baoding (++$N_{rollouts}$) average SSL loss ",
    "log_dir": "appendix",
    "experiment_name": "shadow_baoding_dynamics",
    "legend_names": [r"$N_{rollouts}=1$", r"$N_{rollouts}=2$", r"$N_{rollouts}=4$"],
    "run_names": ["N=1", "N=2", "N=4"],
    "tag": "aux_loss", #tp_rate@t=0", "tp_rate@t=1", "tp_rate@t=2"], # object_found_easy
    "color": [ppo_colour],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 200,
    "x_min": 0,
    "y_max": 0.05,  
    "y_min": 0.017,
    "legend_loc": "upper right",
    "x_label": "Number of RL rollouts",
    "y_label": "Average SSL loss",
    "output_file": "aux_loss_baoding.pdf"
}

bounce_memory_mean_return = {
    "title": "Bounce",
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop-tactile)", "+full dynamics", r"++$N_{rollouts}$"],
    "run_names": ["e2e", "dynamics_2", "memory"],
    "tag": "mean_eval_return", # object_found_easy
    "color": [ppo_colour, dynamics_colour, memory_colour],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 900,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

bounce_ssl_bounces = {
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop)","PPO(prop-tactile)", "+recon", "+tactile recon", "+dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e", "recon_2", "tactile_recon_2", "dynamics_2", "tactile_dynamics"],
    "tag": "bounce_reward", # object_found_easy
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour,  dynamics_colour, dynamics_colour,  "tab:purple"],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 85,  
    "y_min": 0,
    "legend_loc": "lower right",
    "title": r"Number of bounces",
    "output_file": "num_bounces.pdf"

}

bounce_tactile_preds = {
    "title": "Bounce",
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["+tactile recon", "+dynamics", "+tactile dynamics"],
    "run_names": ["tactile_dynamics"],
    "tag": "true_positive_weight @t=0", # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour,  dynamics_colour, dynamics_colour, "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-', '-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 900,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
    "output_file": "pred.pdf",

}

baoding_ssl_rotations = {
    "title": "Number of Baoding rotations (all seeds)",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon", "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_3", "full_recon", "tactile_recon", "dynamics", "tactile_dynamics"],
    "tag": "num_rotations",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour,  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 17,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Complete rotations",
    "output_file": "num_rotations.pdf"
}

baoding_ssl_rotations_minimal = {
    "title": "Number of Baoding rotations (minimal, all seeds)",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+tactile recon", "+full dynamics"],
    "run_names": ["prop", "e2e_3", "tactile_recon", "dynamics"],
    "tag": "num_rotations",
    "color": [prop_colour, ppo_colour, recon_colour, dynamics_colour, dynamics_colour,  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','--', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 17,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Complete rotations",
    "output_file": "num_rotations_minimal.pdf"
}

baoding_ssl_rotations_minimal_good = {
    "title": "Number of Baoding rotations (minimal, no bad seeds)",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding_good",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+tactile recon", "+full dynamics"],
    "run_names": ["prop", "e2e_3", "tactile_recon", "dynamics"],
    "tag": "num_rotations",
    "color": [prop_colour, ppo_colour, recon_colour, dynamics_colour, dynamics_colour,  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','--', '-', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 17,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Complete rotations",
    "output_file": "num_rotations_minimal_good.pdf"
}

baoding_ssl_rotations_good = {
    "title": "Number of Baoding rotations (no bad seeds)",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding_good",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon", "+full dynamics", "+tactile dynamics"],
    "run_names": ["prop", "e2e_3", "full_recon", "tactile_recon", "dynamics", "tactile_dynamics"],
    "tag": "num_rotations",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour,  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 17,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Complete rotations",
    "output_file": "num_rotations_good.pdf"
}

baoding_ssl_rotations_barchart = {
    "title": "Mean and maximum complete rotations in Baoding",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tac recon", "+full dyn", "+memory"],
    "run_names": ["prop", "e2e_3",  "full_recon", "tactile_recon", "dynamics", "memory"],
    "tag": "num_rotations",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, memory_colour],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "bar_chart": True,
    "x_max": 199,
    "x_min": 0,
    "y_max": 25,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Maximum complete rotations",
    "output_file": "num_rotations_barchart.pdf"
}

bounce_ssl_rotations_barchart = {
    "title": "Mean and maximum complete rotations in Baoding",
    "log_dir": "part_2",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop)","PPO(prop-tactile)", "+recon", "+tactile recon", "+dynamics", "memory"],
    "run_names": ["prop", "e2e", "recon_2", "tactile_recon_2", "dynamics_2", "memory"],
    "tag": "bounce_reward",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, memory_colour],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "bar_chart": True,
    "y_max": 90,  
    "y_min": 50,
    "legend_loc": "upper left",
    "y_label": "Maximum complete rotations",
    "output_file": "num_bounces_barchart.pdf"
}

franka_barchart = {
    "title": "Mean and maximum complete rotations in Baoding",
    "log_dir": "part_2",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop)", "PPO(prop-tactile)", "+full recon", "+tactile recon",  "+full dynamics", "+memory"],
    "run_names": ["prop", "e2e_2", "recon_2", "tactile_recon_2", "dynamics_2", "memory"],
    "tag": "timesteps_to_find_object_med",
    "color": [prop_colour, ppo_colour, recon_colour, recon_colour, dynamics_colour, memory_colour],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "bar_chart": True,
    "y_max": 3.9,  
    "y_min": 2.7,
    "legend_loc": "upper left",
    "y_label": "Maximum complete rotations",
    "output_file": "franka_barchart.pdf"
}


baoding_ssl_mean_return = {
    "title": "Baoding",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO", "PPO(prop)", "+full recon", "+tactile recon", "+full dynamics", "+tactile dynamics"],
    "run_names": ["e2e_3", "prop", "full_recon", "tactile_recon", "dynamics", "tactile_dynamics"],
    "tag": "mean_eval_return",  # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "purple",  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 420,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

baoding_aux_loss = {
    "title": "Baoding",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding_dynamics",
    "legend_names": ["N=1", "N=2"],
    "run_names": ["N=1", "N=2"],
    "tag": "Aux loss",  # object_found_easy
    "color": [ppo_colour, prop_colour, recon_colour, recon_colour, dynamics_colour, dynamics_colour, "purple",  "tab:red", "tab:green", "tab:green", "tab:red", "tab:red", "tab:purple", "tab:brown", "tab:blue", "tab:red", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 0.5,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
    "output_file": "aux_loss.png"
}

baoding_memory_mean_return = {
    "title": "Baoding",
    "log_dir": "part_2",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "+full dynamics", "N_rollouts"],
    "run_names": ["e2e_3", "dynamics", "memory"],
    "tag": "mean_eval_return",  # object_found_easy
    "color": [ppo_colour, dynamics_colour, memory_colour],
    "linestyle": ['-', '-','-', '--', "-", "--", '-', '--', '-', '--', "-", "--", '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 500,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Mean evaluation return",
}

franka_tactile_weight = {
    "title": "Find",
    "log_dir": "rebuttal",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 7,  
    "y_min": 4,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

baoding_tactile_weight = {
    "title": "Baoding",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 100,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

bounce_tactile_weight = {
    "title": "Bounce",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 50,  
    "y_min": 0,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

franka_tactile_jacobian = {
    "title": "Find",
    "log_dir": "rebuttal",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 300,  
    "y_min": 4,
    "legend_loc": "upper left",
    "y_label": "Tactile jacobian",
}

baoding_tactile_jacobian = {
    "title": "Baoding",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 400,  
    "y_min": 100,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

bounce_tactile_jacobian = {
    "title": "Bounce",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "tactile_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 750,  
    "y_min": 50,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

franka_prop_jacobian = {
    "title": "Find",
    "log_dir": "rebuttal",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 650,  
    "y_min": 200,
}

baoding_prop_jacobian = {
    "title": "Baoding",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 400,  
    "y_min": 100,
}

bounce_prop_jacobian = {
    "title": "Bounce",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_jacobian",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 650,  
    "y_min": 50,
}

franka_prop_weight = {
    "title": "Find",
    "log_dir": "rebuttal",
    "experiment_name": "franka_find",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 23,  
    "y_min": 18,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

baoding_prop_weight = {
    "title": "Baoding",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_baoding",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 100,  
    "y_min": 18,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}

bounce_prop_weight = {
    "title": "Bounce",
    "log_dir": "rebuttal",
    "experiment_name": "shadow_bounce",
    "legend_names": ["PPO(prop-tactile)", "+tactile recon"],
    "run_names": ["prop_tactile", "tac_recon"],
    "tag": "prop_weight_norm",  # object_found_easy
    "color": [ppo_colour, recon_colour],
    "linestyle": ['-', '--'],
    "max_reward": 0,
    "x_max": 199,
    "x_min": 0,
    "y_max": 35,  
    "y_min": 15,
    "legend_loc": "upper left",
    "y_label": "Tactile weight norm",
}


combo_plot = {
    "exp_list": [franka_prop_stack, bounce_prop_stack, baoding_prop_stack],
    "output_file": "prop.pdf",
    "y_label": "Mean evaluation return",
    "legend_loc": "lower right",
}


study_plot = {
    "exp_list": [franka_study_mean_return, bounce_study, baoding_study_mean_return],
    "output_file": "studying.pdf",
    "y_label": "Mean evaluation return",
    "legend_loc": "upper left",
    "opacity": 0.08,
    "legend_idx": 2
}


ssl_plot = {
    "exp_list": [franka_ssl_mean_return, bounce_ssl_mean_return, baoding_ssl_mean_return],
    "output_file": "ssl_mean_return_plot.pdf",
    "y_label": "Mean evaluation return",
    "legend_loc": "lower right",
    "opacity": 0.08,
    "legend_idx": 0
}

memory_plot = {
    "exp_list": [franka_memory_mean_return, bounce_memory_mean_return, baoding_memory_mean_return],
    "output_file": "memory_mean_return_plot.pdf",
    "y_label": "Mean evaluation return",
    "legend_loc": "lower right",
    "opacity": 0.08,
    "legend_idx": 1
}

franka_extra_plot = {
    "exp_list": [franka_ssl_timesteps_easy, franka_ssl_timesteps, franka_ssl_timesteps_hard],
    "output_file": "franka_extra_plot.pdf",
    "y_label": "Seconds to find object",
    "legend_loc": "upper right",
    "opacity": 0.08,
    "legend_idx": 0
}
  
tactile_weight = {
    "exp_list": [franka_tactile_weight, bounce_tactile_weight, baoding_tactile_weight],
    "output_file": "tactile_weight.png",
    "y_label": "tactile_weight",
    "legend_loc": "lower right",
    "opacity": 0.08,
    "legend_idx": 1
}

tactile_jacobian = {
    "exp_list": [franka_tactile_jacobian, bounce_tactile_jacobian, baoding_tactile_jacobian],
    "output_file": "tactile_jacobian.png",
    "y_label": "tactile_jacobian",
    "legend_loc": "lower right",
    "opacity": 0.01,
    "legend_idx": 1
}

prop_weight = {
    "exp_list": [franka_prop_weight, bounce_prop_weight, baoding_prop_weight],
    "output_file": "prop_weight.png",
    "y_label": "prop weight",
    "legend_loc": "lower right",
    "opacity": 0.08,
    "legend_idx": 1
}
   
prop_jacobian = {
    "exp_list": [franka_prop_jacobian, bounce_prop_jacobian, baoding_prop_jacobian],
    "output_file": "prop_jacobian.png",
    "y_label": "prop_jacobian",
    "legend_loc": "lower right",
    "opacity": 0.01,
    "legend_idx": 1
}
   