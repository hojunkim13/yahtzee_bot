import random as rand
import pprint as pp
from collections import defaultdict

# =============================================================================
#                       SCORING
# =============================================================================
# Predicates assume the hand has been sorted. 
def has_straight(h,n):
    if n==4:
        return h[:-1] == [d for d in range(h[0],h[0]+4)] or h[1:] == [d for d in range(h[1],h[1]+4)]
    elif n==5:
        return h == [d for d in range(h[0],h[0]+5)]
    else:
        return False

def has_small_straight(h):
    return has_straight(h,4)

def count_small_straight(h):
    return 30 if has_small_straight(h) else 0

def has_large_straight(h):
    return has_straight(h,5)

def count_large_straight(h):
    return 40 if has_large_straight(h) else 0

def has_n_of_a_kind(h,n):
    c = defaultdict(int)
    for d in h:
        c[d] += 1
    ns = [d for (d,v) in c.items() if v==n]
    return None if len(ns) == 0 else ns[0]

def has_3_of_a_kind(h):
    return has_n_of_a_kind(h,3)

def count_3_of_a_kind(h):
    res = has_3_of_a_kind(h)
    return 3*res if res!=None else 0

def has_4_of_a_kind(h):
    return has_n_of_a_kind(h,4)

def count_4_of_a_kind(h):
    res = has_4_of_a_kind(h)
    return 4*res if res!=None else 0

def has_yhatzee(h):
    return has_n_of_a_kind(h,5)

def count_yhatzee(h):
    return 50 if has_yhatzee(h) else 0
    
def has_full_house(h):
    d = defaultdict(int)
    for c in h:
        d[c] = d[c]+1
    if len([c for (c,n) in d.items() if n==3]) == 1 and len([c for (c,n) in d.items() if n==2]) == 1:
        return True
    else:
        False

def count_full_house(h):
    return 25 if has_full_house(h) else 0

def count_ns(h,n):
    return sum([v for v in h if v==n])

def count_chance(h):
    return sum(h)

def game_done(scorecard):
    return len([v for v in scorecard.values() if v!=None]) == 13

# @TODO: Additional Yhatzees.
def build_scorecard():
    scorecard = { 'ones':              None
                , 'twos':              None
                , 'threes':            None
                , 'fours':             None
                , 'fives':             None
                , 'sixes':             None
                , 'bonus':             None
                , 'threeofakind':      None
                , 'fourofakind':       None
                , 'smallstraight':     None
                , 'largestraight':     None
                , 'fullhouse':         None
                , 'yhatzee':           None
                , 'chance':            None
                }
    return scorecard

score_map = { 'ones':              lambda h: count_ns(h,1)
            , 'twos':              lambda h: count_ns(h,2)
            , 'threes':            lambda h: count_ns(h,3)
            , 'fours':             lambda h: count_ns(h,4)
            , 'fives':             lambda h: count_ns(h,5)
            , 'sixes':             lambda h: count_ns(h,6)
            , 'threeofakind':      count_3_of_a_kind
            , 'fourofakind':       count_4_of_a_kind
            , 'smallstraight':     count_small_straight
            , 'largestraight':     count_large_straight
            , 'fullhouse':         count_full_house
            , 'yhatzee':           count_yhatzee
            , 'chance':            count_chance
            }

def get_possible_options(h, scorecard):
    return [o for (o,s) in scorecard.items() if s==None and o!='bonus']

def mark_scorecard(h,scorecard,user_choice):
    scorecard[user_choice] = score_map[user_choice](h)

def total_top(scorecard):
    to_count = ['ones','twos','threes','fours','fives','sixes']
    return sum([scorecard[v] for v in to_count])

def total_points(scorecard):
    total = sum([v for (o,v) in scorecard.items() if v !=None])
    top_total = total_top(scorecard)
    return total+35 if top_total>63 else total

# =============================================================================
#                          DICE ROLLING
# =============================================================================
def roll_n_dice(n):
    return sorted([rand.randint(1,6) for _ in range(n)])

# =============================================================================
#                       USER I/O INTERACTIONS
# =============================================================================
def get_roll_input():
    c = input('Press (r) to roll\n')
    while c != 'r':
        print('Press (r)! not (%s)' % c)
        c = input('Press (r) to roll\n')

def select_dice_to_keep(h):
    flag_string = input('Select which dice to keep.\n')
    keep = [d for i,d in enumerate(h) if flag_string[i]=='k']
    return sorted(keep)

def perform_roll():
    # Need to add more logic to this method so that it's actually correct.
    num_dice = 5
    current_hand = []

    # Perform the intial roll 
    get_roll_input()
    current_roll = roll_n_dice(num_dice)
    display_roll(current_roll)
    current_roll = select_dice_to_keep(current_roll)

    # @TODO: Factor out some of this logic.
    if len(current_roll) < 5:
        get_roll_input()
        num_dice = 5-len(current_roll)
        new_dice = roll_n_dice(num_dice)
        current_roll = current_roll+new_dice
        display_roll(current_roll)
        current_roll = select_dice_to_keep(current_roll)

        if len(current_roll) < 5:
            get_roll_input()
            num_dice = 5-len(current_roll)
            new_dice = roll_n_dice(num_dice)
            current_roll = current_roll+new_dice
            display_roll(current_roll)

    h = sorted(current_roll)
    return h

def display_options(opts):
    for i,o in enumerate(opts):
        print('[%d] - %s' % (i+1,o))

def get_choice(opts):
    c = int(input('Select an option\n'))
    while c <= 0 or c > len(opts):
        print('Make a valid selection!')
        c = int(input('Select an option\n'))
    return c

def get_user_decision(h, scorecard):
    opts = get_possible_options(h,scorecard) 
    display_options(opts)
    user_choice = get_choice(opts)
    return opts[user_choice-1]

def print_dice(roll):
    # I want this to print the following: 
    # 
    # ------- ------- ------- ------- ------- ------- 
    # |     | | o   | | o   | | o o | | o o | | o o | 
    # |  o  | |     | |  o  | |     | |  o  | | o o | 
    # |     | |   o | |   o | | o o | | o o | | o o | 
    # ------- ------- ------- ------- ------- ------- 
    # 
    # For 1..6 respectively.
    dot_map = { 6: [(1,2),(1,4),(2,2),(2,4),(3,2),(3,4)]
              , 5: [(1,2),(1,4),(2,3),(3,2),(3,4)]
              , 4: [(1,2),(1,4),(3,2),(3,4)]
              , 3: [(1,2),(2,3),(3,4)]
              , 2: [(1,2),(3,4)]
              , 1: [(2,3)]
              }
    for i in range(5):
        for j in range(8*len(roll)):
            if (i==0 or i==4) and j%8!=7:
                print('-',end='')
            elif 0<i<4 and (j%8==0 or j%8==6):
                print('|',end='')
            elif (i,j%8) in dot_map[roll[j//8]]:
                print('o',end='')
            else:
                print(' ',end='')
        print('')

def display_roll(h):
    print('You rolled: ')
    print_dice(h)

def display_scorecard(sc):
    pp.pprint({k: v for k,v in sc.items() if v!=None})

def input_loop(scorecard):
    # (1) Roll
    # (2) Decide on what to score.
    # (3) Tally Points and update internal state.
    # (4) Repeat.
    while not game_done(scorecard):
        h = perform_roll()
        user_choice = get_user_decision(h,scorecard)
        mark_scorecard(h,scorecard,user_choice)
        display_scorecard(scorecard)
        # pp.pprint(scorecard)
    
    tot_points = total_points(scorecard)
    print('You scored %d points with the following scorecard!!' % tot_points)
    pp.pprint(scorecard)

def main():
    input_loop(build_scorecard())

if __name__ == '__main__':
    main()