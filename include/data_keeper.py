import pandas as pd
import numpy as np

"""
Holds datasets. Returns valid training datasets and testing or validation
sets chosen by given parameters.
"""

class DataKeeper:
    def __init__(self, steps=35, D = 7, 
                 test_k_targets = [0.85, 0.925, 1.0, 1.075, 1.15],
                 test_exp_targets = [10, 30, 60],
                 fname = 'data/train.csv',
                 fname_validation = 'data/validation.csv',
                 fname_test = 'data/test.csv'):
        
        self.steps = steps
        self.D = D
        self.test_k_targets = test_k_targets
        self.test_exp_targets = test_exp_targets
        
        self.even_odds_year = True

        self.train = pd.read_csv(fname)
        self.train_starting_points = self.train[(self.train['nbr_next_steps'] >= self.steps) & (self.train['quote_datetime'].str.slice(11,16) =='09:31') ]['expiration'].str.slice(0,4) 

        self.test, self.test_starting_points = self.prepare_test(fname_test)
        self.validation, self.validation_starting_points = self.prepare_test(fname_validation)
        
        self.test_starting_times = sorted(self.test_starting_points.unique())
        self.validation_starting_times = sorted(self.validation_starting_points.unique())
        
        self.switch_to_test()
        self.reset()
        
    def reset(self, soft=False):
        self.test_cur_day = self.starting_times[0]
        self.day_index = 0
        self.intraday_index = 0
        self.out_of_test_data = False
        self.no_more_sets = False
        
        self.set_index = 0
        
        self.s = 0
        self.f = 0
        
        self.set_test_date(0)
        if not soft:
            self.create_good_sets()
        
    def prepare_test(self, fname):
        test = pd.read_csv(fname)
        
        test['scaled K'] = test['strike'] / test['underlying_bid']
        test['dtm'] = (pd.to_datetime(test['expiration']) + pd.DateOffset(hours=15, minutes=31) - pd.to_datetime(test['quote_datetime'])) /np.timedelta64(1, 'D')
        
        test_starting_points = test[(test['nbr_next_steps'] >= self.steps) \
                                            & (test['quote_datetime'].str.slice(11,16) =='09:31') \
                                            & (test['dtm'] > (self.steps / self.D + 1))]['quote_datetime']
        
        return test, test_starting_points

        
    def next_train_set(self):
        start = np.random.choice(self.train_starting_points.index)
        
        next_set = self.train.loc[start:start + self.steps, :]
        next_set = next_set.reset_index(drop = True)

        divisor = next_set.loc[0, 'underlying_bid']
        for key in ['underlying_bid','underlying_ask','bid','ask','strike']:
            next_set[key] = next_set[key] / divisor
        
        return next_set
    
    def set_test_date(self, idx):
        self.day_index = idx
        self.intraday_index = 0
        
        if self.day_index < len(self.starting_times):
            self.test_cur_day = self.starting_times[self.day_index]
            self.out_of_test_data = False
        else:
            self.out_of_test_data = True
            
        return self.out_of_test_data
    
    def switch_to_validation(self):
        self.dataset = self.validation
        self.starting_points = self.validation_starting_points
        self.starting_times = self.validation_starting_times
        
    def switch_to_test(self):
        self.dataset = self.test
        self.starting_points = self.test_starting_points
        self.starting_times = self.test_starting_times
    
    def next_test_set(self):
        
        if self.set_index == self.set_count:
            return None
        
        self.set_index += 1
        if self.set_index == self.set_count:
            self.no_more_sets = True
            
        return self.good_sets[self.set_index - 1]
        
    def create_good_sets(self):
        
        self.good_sets = []
        
        while not self.out_of_test_data: 
            df = self.create_set()
            if df is None:
                break
            else:
                self.good_sets.append(df)
                
        self.set_count = len(self.good_sets)
            
    def create_set(self):
        # Creates the next good set by finding an option that is closests to
        # strike and expiry criteria
        not_found = True
        
        while not_found:
            
            k_index = int(np.floor(self.intraday_index / len(self.test_exp_targets)))
            exp_index = self.intraday_index % len(self.test_exp_targets)
            
            k_target = self.test_k_targets[k_index]
            exp_target = self.test_exp_targets[exp_index]
                
            possible_starts = self.starting_points[self.starting_points == self.test_cur_day].index         
            possible_set = self.dataset.loc[possible_starts]
            
            tries = 1
            
            while tries < 11:
                k_range = (k_target - tries * 0.005, k_target + tries * 0.005)
                exp_range = (exp_target - tries * 1, exp_target + tries * 1)
    
                filtered_set = possible_set[(possible_set['scaled K'] >= k_range[0]) & \
                                            (possible_set['scaled K'] <= k_range[1]) & \
                                            (possible_set['dtm'] >= exp_range[0]) & \
                                            (possible_set['dtm'] <= exp_range[1])]
                
                if filtered_set.empty:
                    tries += 1
                else:
                    ranks = abs(filtered_set.loc[:,'scaled K'] / k_target - 1) \
                                   + abs(filtered_set.loc[:,'dtm'] / exp_target - 1)
                    start = ranks.sort_values().index[0]
                    break
                        
            self.intraday_index += 1
            
            if self.intraday_index >= (len(self.test_k_targets) * len(self.test_exp_targets)):
                self.intraday_index = 0
                self.day_index += int(self.steps / self.D)
                
                if self.day_index < len(self.starting_times):
                    self.test_cur_day = self.starting_times[self.day_index]
                else:
                    self.out_of_test_data = True
                    return None
                    
            if tries == 11:
                self.f += 1
            else:
                self.s +=1
                
                next_set = self.dataset.loc[start:start + self.steps, :]
        
                next_set = next_set.reset_index(drop = True)
                next_set = next_set.drop(columns = ['scaled K', 'dtm'])
                
                divisor = next_set.loc[0, 'underlying_bid']
                
                for key in ['underlying_bid','underlying_ask','bid','ask','strike']:
                    next_set[key] = next_set[key] / divisor
                
                return next_set     
        
    
        
    
