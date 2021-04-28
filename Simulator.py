import numpy as np


def preprosessing(state):
    #5 * 6
    dice_state = [a-1 for a in state["dice"]]
    dice_state = np.eye(6)[dice_state]

    #13 * 6
    empty_mask = list(map(lambda x : 1 if x is None else 0, state["table"]))
    left_table = [6 * [a] for a in empty_mask]

    #1 * 6
    left_rollout = [np.eye(6)[state["left_rollout"]]]

    #1 * 6
    left_turn = min(sum(empty_mask), 5)
    left_turn = [np.eye(6)[left_turn]]
    
    total_state = np.concatenate((dice_state, left_table, left_rollout, left_rollout), axis = 0)
    return total_state


def rolloutDice(dice_state = None, rollout_action = None):
    if dice_state is None:
        return list(np.random.randint(1, 7, 5))
    else:
        dice_state = dice_state.copy()
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
    legal_dice_actions = list(range(31)) if left_rollout else []        
    legal_score_actions = [idx+31 for idx, occupied in enumerate(score_table) if occupied is None]
    legal_actions = legal_dice_actions + legal_score_actions
    return legal_actions

