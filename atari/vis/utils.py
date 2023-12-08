import wandb
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]

####################
# Plotting Helpers

def save_fig(fig, name):
    file_name = '{}.pdf'.format(name)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    return file_name

def set_axes(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=14)

def set_ticks(ax, xticks, xticklabels, yticks, yticklabels):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

def decorate_axis(ax, wrect=10, hrect=10, labelsize='medium'):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    
    
######################
# @title Helpers 
def pgs(f):
    print(inspect.getsource(f))

def score_normalization(res_dict, min_scores, max_scores):
    games = res_dict.keys()
    norm_scores = {}
    for game, scores in res_dict.items():
        norm_scores[game] = (scores - min_scores[game])/(max_scores[game] - min_scores[game])
    return norm_scores

def plot_score_hist(score_matrix, bins=20, figsize=(28, 14), 
                    fontsize='xx-large', N=6, extra_row=1,
                    names=None):
    num_tasks = score_matrix.shape[1]
    if names is None:
        names = ATARI_100K_GAMES
    N1 = (num_tasks // N) + extra_row
    fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
    for i in range(N):
        for j in range(N1):
            idx = j * N + i
            if idx < num_tasks:
                ax[j, i].set_title(names[idx], fontsize=fontsize)
                sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j,i], kde=True)
            else:
                ax[j, i].axis('off')
            decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
            ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
            if idx % N == 0:
                ax[j, i].set_ylabel('Count', size=fontsize)
            else:
                ax[j, i].yaxis.label.set_visible(False)
            ax[j, i].grid(axis='y', alpha=0.1)
    return fig


#######################
# ATARI
RANDOM_SCORES = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}

HUMAN_SCORES = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}

DQN_SCORES = {
    'Alien': 3069,
    'Amidar': 739.5,
    'Assault': 3359,
    'Asterix': 6012,
    'BankHeist': 429.7,
    'BattleZone': 26300.,
    'Boxing': 71.8,
    'Breakout': 401.2,
    'ChopperCommand': 6687.,
    'CrazyClimber': 114103,
    'DemonAttack': 9711.,
    'Freeway': 30.3,
    'Frostbite': 328.3,
    'Gopher': 8520.,
    'Hero': 19950., 
    'Jamesbond': 576.7,
    'Kangaroo': 6740.,
    'Krull': 3805., 
    'KungFuMaster': 23270.,
    'MsPacman': 2311.,
    'Pong': 18.9, 
    'PrivateEye': 1788.,
    'Qbert': 10596.,
    'RoadRunner': 18257., 
    'Seaquest': 5286.,
    'UpNDown': 8456.
}


########################
# Subsampler
def subsample_scores(score_dict, n=5, replace=False):
    subsampled_dict = {}
    total_samples = len(score_dict[list(score_dict.keys())[0]])
    for game, scores in score_dict.items():
        indices = np.random.choice(range(total_samples), size=n, replace=replace)
        subsampled_dict[game] = scores[indices]
    return subsampled_dict

def subsample_scores_mat(score_mat, num_samples=5, replace=False):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    subsampled_scores = np.empty((num_samples, num_games))
    for i in range(num_games):
        indices = np.random.choice(total_samples, size=num_samples, replace=replace)
        subsampled_scores[:, i] = score_mat[indices, i]
    return subsampled_scores

def subsample_seeds(score_mat, num_samples=5, replace=False):
    indices = np.random.choice(
        score_mat.shape[0], size=num_samples, replace=replace)
    return score_mat[indices]

def batch_subsample_seeds(score_mat, num_samples=5, batch_size=100,
                          replace=False):
    indices = [
        np.random.choice(score_mat.shape[0], size=num_samples, replace=replace)
        for _ in range(batch_size)
    ]
    return (score_mat[idx] for idx in indices)

def subsample_scores_mat_with_replacement(score_mat, num_samples=5):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    indices = np.random.choice(
      total_samples, size=(num_samples, num_games), replace=True)
    col_indices =  np.expand_dims(np.arange(num_games), axis=0)
    col_indices = np.repeat(col_indices, num_samples, axis=0)
    subsampled_scores = score_mat[indices, col_indices]
    return subsampled_scores

###########################
# aggregate function

#@title Aggregate computation helpers

SIZES = [3, 5, 10, 25, 50, 100]

def calc_aggregate_fn(score_data, num_samples, total_n, 
                      aggregate_fn, replace):
    subsampled_scores = batch_subsample_seeds(
      score_data, num_samples, batch_size=total_n, replace=replace)
    aggregates = [aggregate_fn(scores) for scores in subsampled_scores]
    return np.array(aggregates)

def calculate_aggregate_varying_sizes(score_matrix, aggregate_fn, total_n=20000,
                                      sizes=None, replace=False):
    agg_dict = {}
    if sizes is None:
        sizes = SIZES
    for size in sizes:
        agg_dict[size] = calc_aggregate_fn(score_matrix, num_samples=size, aggregate_fn=aggregate_fn,  # change n to size
                                    total_n=total_n, replace=replace)
        print('Mean Aggregate: {}'.format(np.mean(agg_dict[size])))
    return agg_dict

def CI(bootstrap_dist, stat_val=None, alpha=0.05, is_pivotal=False):
    """
    Get the bootstrap confidence interval for a given distribution.
    Args:
      bootstrap_distribution: numpy array of bootstrap results.
      stat_val: The overall statistic that this method is attempting to
        calculate error bars for. Default is None.
      alpha: The alpha value for the confidence intervals.
      is_pivotal: if true, use the pivotal (reverse percentile) method. 
        If false, use the percentile method.
    Returns:
      (low, high): The lower and upper limit for `alpha` x 100% CIs.
      val: The median value of the bootstrap distribution if `stat_val` is None
        else `stat_val`.
    """
    # Adapted from https://pypi.org/project/bootstrapped
    if is_pivotal:
        assert stat_val is not None, 'Please pass the statistic for a pivotal'
        'confidence interval' 
        low = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
        val = stat_val
        high = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    else:
        low = np.percentile(bootstrap_dist, 100 * (alpha / 2.))
        val = np.percentile(bootstrap_dist, 50)
        high = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
    return (low, high), val

######################
# Wandb
def collect_runs(project_name, filters=None):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_name, filters=filters)
    summary_list, group_list, config_list, id_list = [], [], [], []
    for run in tqdm(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        
        group_list.append(run.group)
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
              if not k.startswith('_')})

        # .name is the human-readable name of the run.
        id_list.append(run.id)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "group": group_list,
        "config": config_list,
        "id": id_list,
        })
    
    return runs_df

def filter_runs(runs_df, exp_name, group_name):
    configs = runs_df['config']
    indexs = []
    for idx, config in enumerate(configs):
        if len(config) == 0:
            continue

        run_exp_name = config['exp_name']
        run_group_name = config['group_name']

        # condition
        if run_exp_name == exp_name and run_group_name == group_name:
            if 'env' in config:
                indexs.append(idx)
                
    data = runs_df.iloc[indexs]
    
    return data

def get_scores(data, metric='eval_mean_traj_game_scores'):
    scores = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        summary = row['summary']
        config = row['config']

        if 'env' not in config:
            continue

        game = config['env']['game']
        try:
            score = summary[metric]
        except:
            continue
        
        if score != 'NaN':
            scores.append([0, game, score, 0])

    return scores

def snake_to_camel(name):
    return ''.join(word.title() for word in name.split('_'))

def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)
  
def generate_score_matrix(games, scores, camel=True, num_seeds=10):
    _scores = {}
    for game in games:
        _scores[game] = []

    for score in scores:
        if camel:
            game = snake_to_camel(score[1])
        else:
            game = score[1]

        if (game == 'Median') or (game == 'Mean'):
            continue
        _scores[game].append(score[2])
            
    num_min_seed = 999
    for game, score in _scores.items():
        num_min_seed = min(num_min_seed, len(score))
    
    num_min_seed = min(num_min_seed, num_seeds)
    for game, score in _scores.items():
        _scores[game] = np.array(score[-num_min_seed:])

    print(_scores)
    raw_score_matrix = convert_to_matrix(_scores)
    scores = score_normalization(_scores, RANDOM_SCORES, HUMAN_SCORES)
    score_matrix = convert_to_matrix(scores)
        
    return raw_score_matrix, scores, score_matrix