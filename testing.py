import numpy as np
import pandas as pd
import os
import argparse
from include.settings import getSettings, setSettings
from include.env import Env
from include.utility import StatePrepare, maybe_make_dirs
from include.actor_critic import ActorCritic

def test_run(env, actor_critic, scaler, state_size, episode_number, empirical):
    cur_state = env.reset(empirical)
    cur_state = scaler.transform(cur_state)
    cur_state = cur_state.reshape((1, state_size))
    infos = None
    
    stats = {'rewards':np.zeros(1), 'b rewards':np.zeros(1), 'pnl':np.zeros(1), 'b pnl':np.zeros(1)}
    i = 0

    done = False
    while not done:
        pred_action = actor_critic.act(cur_state)
        action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1) 

        new_state, reward, done, info = env.step(action)
        info['episode'] = episode_number
        info['cr1'] = 0
        info['cr2'] = 0
        info['ticker'] = env.ticker
        
        if infos is None:
            infos = pd.DataFrame(info, index=[0])
        else:
            infos = infos.append(info, ignore_index=True)
        
        new_state = scaler.transform(new_state) 
        new_state = new_state.reshape((1, state_size))
        cur_state = new_state
        
        stats['rewards'][i] += reward
        stats['b rewards'][i] += info['B Reward']
        stats['pnl'][i] += info['A PnL']
        stats['b pnl'][i] += info['B PnL']
    return stats, episode_number, infos

def test_load(model_name):
    maybe_make_dirs()

    model_name, i = model_name.rsplit('_', 1)
    sim = 1000

    setSettings(model_name)
    s = getSettings()
    env = Env(s)

    empirical = s['process'] == 'Real'

    scaler = StatePrepare(env, 1, model_name)
    scaler.load(model_name)
    state_size = scaler.state_size
    
    actor_critic = ActorCritic(state_size)
    actor_critic.load('model/' + model_name + '_' + i)

    folder = 'results/testing/'
    output_filename = folder + model_name + '_' + i + '.csv'
    
    a_rewards = 0
    b_rewards = 0
    j = 0
    info_df = None
    
    if empirical:
        env.data_keeper.switch_to_test()
        env.data_keeper.reset(soft=False)
        set_count = env.data_keeper.set_count
        while not env.data_keeper.no_more_sets:
            stats, _, t_info = test_run(env, actor_critic, scaler, state_size, j, empirical)
            a_rewards += np.sum(stats['rewards'])
            b_rewards += np.sum(stats['b rewards'])
            print("\rEpisode {}/{}".format(j, set_count), end="")
            if info_df is None:
                info_df = t_info
            else:
                info_df = info_df.append(t_info, ignore_index=True)
            j += 1
    else:
        while j < sim:
            stats, _, t_info = test_run(env, actor_critic, scaler, state_size, j, empirical)
            a_rewards += np.sum(stats['rewards'])
            b_rewards += np.sum(stats['b rewards'])
            print("\rEpisode {}/{}".format(j + 1, sim), end="")
            if info_df is None:
                info_df = t_info
            else:
                info_df = info_df.append(t_info, ignore_index=True)
            j += 1

    info_df.to_csv(output_filename)
    print('\nTesting: {:.0f} vs {:.0f}'.format(a_rewards, b_rewards))

def read_validation_files(model):
    folder = 'results/'
    best = -1000
    best_name = ''
    
    directory = os.fsencode(folder)
    for file in os.listdir(directory):
        fname = os.fsdecode(file)
        if model in fname:
            df = pd.read_csv(folder + fname)
            sums = df.sum(axis=0)
            sums = sums.loc[['A Reward','B Reward']]
            diff = sums.loc['A Reward']-sums.loc['B Reward']
            if diff > best: 
                best = diff
                best_name = fname
            print(fname, ':', diff)

    best_name = best_name.replace('.csv', '')
    print(f'Best: {best_name}, {best}')
    
def result_eval(model):
    fn = f'results/testing/{model}'
    if '.csv' not in fn:
        fn += '.csv'
        
    e = pd.read_csv(fn).groupby(['episode']).sum()
    val = 100*e[['A PnL', 'B PnL']].mean()
    print('Mean   | {:7.4f} | {:7.4f}'.format(*val))
    val = 100*e[['A TC','B TC']].mean()
    print('Mean T | {:7.4f} | {:7.4f}'.format(*val))
    val = 100*e[['A PnL', 'B PnL']].std()
    print('std    | {:7.4f} | {:7.4f}'.format(*val))    
    val =  e[['A Reward', 'B Reward']].mean()
    print('Reward | {:7.2f} | {:7.2f}'.format(*val))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_argument_group('mode')
    g.add_argument('--test', action='store_true')
    g.add_argument('--validate', action='store_true')
    g.add_argument('--results', action='store_true')
    p.add_argument('--model', required=True)
    args = vars(p.parse_args())
    if args['test']:
        print(f'Testing model {args["model"]}')
        test_load(args['model'])
        result_eval(args['model'])
    elif args['validate']:
        print(f'Validating model {args["model"]}')
        read_validation_files(args['model'])
    elif args['results']:
        print(f'Result eval model {args["model"]}')
        result_eval(args['model'])