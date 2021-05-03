from copy import deepcopy
import numpy as np

input2int = {k: k+31 for k in range(13)}
int2input = {v:k for k,v in input2int.items()}
rollout2int = {            
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
int2rollout = {v:k for k,v in rollout2int.items()}

expectation = [1.88, 5.28, 8.57, 12.16, 15.69, 19.19,
                21.66, 13.10, 22.59,
                29.46, 32.71, 22.01, 16.87]


def preprocessing(state):
    dice_state = [a-1 for a in state["dice"]]
    dice_state = np.eye(6)[dice_state].sum(axis=0).astype(int)
    dice_state = dice_state / 5
    
    reward_table = calcScoretable(state["dice"], state["table"])    
    reward_table = np.array(reward_table) / 50.

    left_rollout = np.eye(3)[state["left_rollout"]]

    turn = 13 - state["table"].count(None)
    turn = np.eye(13)[turn]

    bonus = np.array((state["upper_bonus"], state["yat_bonus"]), dtype = int)
    
    total_state = np.concatenate((dice_state, reward_table, left_rollout, turn, bonus), axis = -1)
    return total_state


def rolloutDice(dice_state = None, rollout_action = None):
    if dice_state is None:
        return list(np.random.randint(1, 7, 5))
    else:
        dice_state = deepcopy(dice_state)
        for index in rollout_action:
            dice_state[index-1] = np.random.randint(1, 7)
    return dice_state

def calcScoretable(dice_state, score_table = None):
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
    if score_table is not None:
        for idx in range(13):
            if score_table[idx] is not None:
                score_mask[idx] = 0        
    return score_mask


def getLegalAction(left_rollout, score_table):
    legal_score_actions = [idx+31 for idx, occupied in enumerate(score_table) if occupied is None]
    if len(legal_score_actions) == 0:
        return []
    else:
        legal_dice_actions = list(range(31)) if left_rollout else []        
        legal_actions = legal_dice_actions + legal_score_actions
        return legal_actions

def calcOutcome(state):
    score = sum(state["table"]) / 100
    if state["upper_bonus"]:
        score += .35
    if state["yat_bonus"]:
        score += 1
    return score


def step_with_int(state_dict, int_action):
    turn_end = False
    if int_action < 31:
        action = int2rollout[int_action]
        action = eval(action)
    else:
        action = int2input[int_action]
    state_dict = deepcopy(state_dict)

    dice_state = state_dict["dice"]
    left_rollout = state_dict["left_rollout"]
    score_table = state_dict["table"]
    upper_bonus = state_dict["upper_bonus"]
    yat_bonus = state_dict["yat_bonus"]
    
    reward = 0
    if type(action) in (tuple, set):
        rollout = list(set(action))
        dice_state = rolloutDice(dice_state, rollout)
        left_rollout -= 1            
    elif type(action) is int:        
        value = calcScoretable(dice_state, score_table)[action]
        score_table[action] = value
        reward += value
            
        #reset Turn
        turn_end = True
        left_rollout = 2
        dice_state = rolloutDice()
    else:
        raise ValueError("INVAILD ACTION")
        
    
    if not upper_bonus and sum([a for a in score_table[:6] if a is not None]) >= 63:
        reward += 35
        upper_bonus = True
    if not yat_bonus and score_table[-1] == 50:
        reward += 100
        yat_bonus = True
    
    state = {
        "dice" : dice_state,
        "table" : score_table,
        "left_rollout" : left_rollout,
        "upper_bonus" : upper_bonus,
        "yat_bonus" : yat_bonus
        }
    done = False if None in score_table else True
    return state, reward, done, turn_end