import numpy as np
import pandas as pd
from os import path
import argparse

from include.env import Env
from testing import test_run

from include.actor_critic import ActorCritic
from include.utility import StatePrepare, get_model_number, maybe_make_dirs
from include.settings import getSettings, saveSettings, setSettings


def main(model_name = None):
    maybe_make_dirs()
    if model_name is None:
        settings = getSettings()
        process = settings['process']
        model_name = process + str(get_model_number(process))
        saveSettings(model_name)
    else:
        setSettings(model_name)
        settings = getSettings()
        process = settings['process']
    print(f'Training model {model_name}')
    
    n_steps = settings['n_steps']
    showcase_every = settings['showcase_every']
    min_noise = settings['min_noise']
    max_noise = settings['max_noise']
    noise_reward_dividor = settings['noise_reward_dividor']

    env = Env(settings)
    if process == 'Real':
        env.data_keeper.switch_to_validation()
        env.data_keeper.reset()

    # agent's and black-scholes hedge's starting positions in the underlying
    start_a, start_b = 0.0, 0.0
    validation_diff = -100000
    validation_n = 0
    current_noise = 1.0

    # feature scaling, save for later use
    scaler = StatePrepare(env, 1, model_name)
    scaler.save()
    state_size = scaler.state_size
    
    actor_critic = ActorCritic(state_size)
    actor_critic.forget()
    
    num_episodes = settings['num_episodes']
    
    show_example = False
    
    stats = {'rewards':np.zeros(num_episodes), 'b rewards':np.zeros(num_episodes), 'pnl':np.zeros(num_episodes), 'b pnl':np.zeros(num_episodes)}
    
    for i in range(num_episodes):
        j = 0
        done = False        

        cur_state = env.reset(False, start_a, start_b)
        cur_state = scaler.transform(cur_state).reshape((1, state_size))
        
        while not done:
            # Show every 'showcase_every'th episode, no noise
            if i % showcase_every == 0:
                show_example = True
            if show_example:
                pred_action = actor_critic.act(cur_state)
                action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1)
                bs_delta = env.get_bs_delta()
                
                if not j: print('\nAgent |  BS  |  Diff')
                print('{:5.2f} |{:5.2f} | {:5.2f}'.format(action, bs_delta, action - bs_delta))
            else:
                noise = [np.random.normal(0, current_noise)]
                pred_action = actor_critic.act(cur_state) + noise
                pred_action = np.clip(pred_action, -1, 1)
                # activation gives [-1, 1], scale to [0, 1] and clip if noise brings over that
                # Pred_action kept as is in the memory
                action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1)

            new_state, reward, done, info = env.step(action)            
            new_state = scaler.transform(new_state)         
            new_state = new_state.reshape((1, state_size))
            actor_critic.remember(cur_state, pred_action, reward, new_state, done)
            
            cur_state = new_state
            
            j += 1
            if j == n_steps:
                show_example = False
                
            # Save interesting stats of the episode
            stats['rewards'][i] += reward
            stats['b rewards'][i] += info['B Reward']
            stats['pnl'][i] += info['A PnL']
            stats['b pnl'][i] += info['B PnL']

        actor_critic.train()

        if i % 100 == 0 and i >= 100:
            ag = np.mean(stats['rewards'][i - 100:i])
            b = np.mean(stats['b rewards'][i - 100:i])
            current_noise = np.clip((b - ag)/noise_reward_dividor, min_noise, max_noise)
            print('{}/{} Last 100: {:2.4} vs {:2.4}, noise: {:2.4}'.format(i+1, num_episodes, ag, b, current_noise))

        if i % settings['validation_interval'] == 0 and i > 0:
            if process == 'Real':
                env.data_keeper.reset(soft=True)
                env.data_keeper.set_test_date(0)
                set_count = env.data_keeper.set_count
            else:
                set_count = settings['sim_test_runs']
            
            actor_critic.save('model/' + model_name + '_' + str(i))
            
            a_rewards = 0
            b_rewards = 0
            
            info_df = None
            k = 0
            while k < set_count:
                test_stats, _, t_info = test_run(env, actor_critic, scaler, state_size, k, process == 'Real')
                a_rewards += np.sum(test_stats['rewards'])
                b_rewards += np.sum(test_stats['b rewards'])
                print("\rValidating Episode {}/{}".format(k + 1, set_count), end="")
                
                if info_df is None:
                    info_df = pd.DataFrame(t_info)
                else:
                    info_df = info_df.append(t_info, ignore_index=True)
                
                k += 1
            
            info_df.to_csv('results/' + model_name + '_' + str(i) + '.csv')
            diff = a_rewards - b_rewards
            
            print('\nValidation: {:.0f} vs {:.0f}'.format(a_rewards, b_rewards))
            
            if diff > validation_diff:
                validation_diff = diff
                validation_n = 0
            else:
                validation_n += 1
            
            if validation_n >= settings['validation_limit']:
                break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_argument_group('mode')
    p.add_argument('--settings', required=False, default=None)
    args = vars(p.parse_args())
    main(args['settings'])