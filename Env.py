import numpy as np

class Yahtzee:
    def __init__(self):
        self.rollout2int = {            
            "{1}": 0,
            "{2}": 1,
            "{3}": 2,
            "{4}": 3,
            "{5}": 4,
            "{1,2}": 5,
            "{1,3}": 6,
            "{1,4}": 7,
            "{1,5}": 8,
            "{2,3}": 9,
            "{2,4}": 10,                        
            "{2,5}": 11,                        
            "{3,4}": 12,                        
            "{3,5}": 13,                        
            "{4,5}": 14,
            "{1,2,3}": 15,
            "{1,2,4}": 16,
            "{1,2,5}": 17,
            "{1,3,4}": 18,
            "{1,3,5}": 19,
            "{1,4,5}": 20,
            "{2,3,4}": 21,
            "{2,3,5}": 22,
            "{2,4,5}": 23,
            "{3,4,5}": 24,
            "{1,2,3,4}" : 25,
            "{1,2,3,5}" : 26,
            "{1,2,4,5}" : 27,
            "{1,3,4,5}" : 28,
            "{2,3,4,5}" : 29,
            "{1,2,3,4,5}" : 30,
            }        
        self.int2rollout = {v:k for k,v in self.rollout2int.items()}
        self.input2int = {k: k+31 for k in range(13)}
        self.int2input = {v:k for k,v in self.input2int.items()}
        
    def reset(self):
        self.dice_state = self.rolloutDice()
        self.score_table = [None] * 13
        self.left_rollout = 2
        self.upper_bonus = 0
        self.yahtzee_bonus = 0        
        state = {
            "dice" : self.dice_state.copy(),
            "table" : self.score_table.copy(),
            "left_rollout" : self.left_rollout,
            "upper_bonus" : self.upper_bonus,
            "yat_bonus" : self.yahtzee_bonus,
            }
        return state
        
    def rolloutDice(self, dice_state = None, rollout_action = None):
        if dice_state is None:
            return list(np.random.randint(1, 7, 5))
        else:
            dice_state = dice_state.copy()
            for index in rollout_action:
                dice_state[index-1] = np.random.randint(1, 7)
        return dice_state
    
    def calcScoretable(self, dice_state = None):
        if dice_state is None:
            dice_state = self.dice_state
        '''
        Return score mask
        '''
        
        #upper section
        aces = dice_state.count(1) 
        twos = dice_state.count(2) * 2
        threes = dice_state.count(3) * 3
        fours = dice_state.count(4) * 4
        fives = dice_state.count(5) * 5
        sixes = dice_state.count(6) * 6
        
        upper_score = [aces, twos, threes, fours, fives, sixes]
        
        #lower section
        uniques = set(dice_state)
        counts = [dice_state.count(n) for n in range(1,7)]

        triple = sum(dice_state) if max(counts) >= 3 else 0
        four_card = sum(dice_state) if max(counts) >= 4 else 0
        full_house = 25 if len(uniques) <= 2 else 0
        small_straight = 30 if len(uniques) >= 4 and (max(uniques) - min(uniques) == 3) else 0
        large_straight = 40 if len(uniques) >= 5 and (max(uniques) - min(uniques) == 4) else 0
        chance = sum(dice_state)
        yahtzee = 50 if len(uniques) == 1 else 0

        lower_score = [triple, four_card, full_house, small_straight,
                        large_straight, chance, yahtzee]

        score_mask = upper_score + lower_score
        for idx in range(13):
            if self.score_table[idx] is not None:
                score_mask[idx] = 0        
        return score_mask


    def getLegalAction(self):
        legal_dice_actions = list(range(31)) if self.left_rollout else []        
        legal_score_actions = [idx+31 for idx, occupied in enumerate(self.score_table) if occupied is None]
        legal_actions = legal_dice_actions + legal_score_actions
        return legal_actions


    def step_with_str(self, action):
        action = eval(action)
        reward = 0
        if type(action) is tuple:        
            rollout = list(set(action))
            self.dice_state = self.rolloutDice(self.dice_state, rollout)
            self.left_rollout -= 1            
        elif type(action) is int:
            if action < 31:
                rollout = [action]
                self.dice_state = self.rolloutDice(self.dice_state, rollout)
                self.left_rollout -= 1
            else:
                table_idx = self.int2input[action]
                value = self.calcScoretable()[table_idx]
                self.score_table[table_idx] = value
                reward += value
                    
                #reset Turn
                self.left_rollout = 2
                self.dice_state = self.rolloutDice()
        else:            
            raise ValueError("INVAILD ACTION")
        
        if not self.upper_bonus and sum([a for a in self.score_table[:6] if a is not None]) >= 63:
            reward += 35
            self.upper_bonus = True
        if not self.yahtzee_bonus and self.score_table[-1] == 50:
            reward += 100
            self.yahtzee_bonus = True
        
        state = {
            "dice" : self.dice_state.copy(),
            "table" : self.score_table.copy(),
            "left_rollout" : self.left_rollout,
            "upper_bonus" : self.upper_bonus,
            "yat_bonus" : self.yahtzee_bonus,
            }
        done = False if None in self.score_table else True
        info = self.left_rollout
        return state, reward, done, info

    def step(self, int_action):        
        if int_action < 31:
            action = self.int2rollout[int_action]
            action = eval(action)
        else:
            action = self.int2input[int_action]
                
        reward = 0
        if type(action) in (tuple, set):
            rollout = list(set(action))
            self.dice_state = self.rolloutDice(self.dice_state, rollout)
            self.left_rollout -= 1            
        elif type(action) is int:        
            value = self.calcScoretable()[action]
            self.score_table[action] = value
            reward += value
                
            #reset Turn
            left_rollout = 2
            dice_state = self.rolloutDice()
        else:
            raise ValueError("INVAILD ACTION")
                    
        if not self.upper_bonus and sum([a for a in self.score_table[:6] if a is not None]) >= 63:
            reward += 35
            self.upper_bonus = True
        if not self.yahtzee_bonus and self.score_table[-1] == 50:
            reward += 100
            self.yahtzee_bonus = True
        
        state = {
            "dice" : self.dice_state.copy(),
            "table" : self.score_table.copy(),
            "left_rollout" : self.left_rollout,
            "upper_bonus" : self.upper_bonus,
            "yat_bonus" : self.yahtzee_bonus,
            }

        done = False if None in self.score_table else True
        info = self.left_rollout
        return state, reward, done, info


    def render(self):
        legal_mask = [1] * 13
        score_mask = self.calcScoretable()
        for idx in range(13):
            if self.score_table[idx] is not None:
                legal_mask[idx] = 0     
        reward_table = [b for a, b in zip(legal_mask, score_mask) if a]
        print("\n\n")
        print("DICE :" ,self.dice_state)        
        print("TABLE :", self.score_table, "SUM :", sum([a for a in self.score_table if a is not None]))
        print("REWARD :", reward_table)
        print("LEFT ROLLOUT :", self.left_rollout)

# def calc_cases(n, r):
#     return np.math.factorial(n) / np.math.factorial(r) / np.math.factorial(n - r)

def gamePlay():    
    env = Yahtzee()
    env.reset()
    done = False        
    score = 0
    while not done:
        env.render()
        legal_actions = env.getLegalAction()
        #action = np.random.choice(legal_actions)
        input_score = [x for x in legal_actions if x >= 31]
        action = input(f"{input_score}\n")
        state, reward, done, info = env.step(action)
        preprocessing(state)
        score += reward
    print(f"Score : {score}")

if __name__ == "__main__":
    from Simulator import preprocessing
    gamePlay()
