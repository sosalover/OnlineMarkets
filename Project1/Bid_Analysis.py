import csv
import random
import matplotlib.pyplot as plt
import numpy as np


def winning_sim(my_value_index):
    with open('bid_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    wins = 0
    ties = 0
    total = 0
    sims = 1000
    for i in range (100000):
        person = random.randint(1, 42)
        person_value = random.randint(0, 9)
        bid = float(data[person][person_value])
        if person == 29:
            continue
        my_value = my_value_index
        my_bid = float(data[29][my_value])
        if my_bid > bid:
            wins +=1
        if my_bid == bid:
            ties +=1
        total += 1
    return((wins + ties/2)/ total)

for i in range(10):
    print(winning_sim(i))

def expected_utility(this_person_number, my_value):
    with open('bid_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    total_utility = 0
    total_runs = 0
    sims = 100000
    for i in range (sims):
        person = random.randint(1, 42)
        person_value = random.randint(0, 9)
        bid = float(data[person][person_value])
        if person == this_person_number:
            continue
       
        my_bid = float(data[this_person_number][my_value])
        if my_bid > bid:
            total_utility += (my_value*10 + 10 - float(my_bid))
        if my_bid == bid:
            coin = random.randint(1,2)
            if coin == 1:
                total_utility += (my_value*10 + 10 - float(my_bid))
        total_runs += 1
    return total_utility/total_runs


'''print("expected utilities")
for i in range(10):
    print(expected_utility(29, i))'''

def find_best(own_value):
    with open('bid_data.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    total_utility = 0
    sims = 1000
    best_bid = -1
    best_utility = 0
    steps = 100
    x_bids = []
    y_utility = []
    for my_bid in range (0,own_value*steps):
        my_bid = float(my_bid/steps)
        total_utility = 0
        for i in range (sims):
            person = random.randint(1, 42)
            person_value = random.randint(0, 9)
            bid = float(data[person][person_value])
            if person == -1:
                continue
            my_value = own_value
            if my_bid > bid:
                total_utility += (own_value - float(my_bid))
            if my_bid == bid:
                coin = random.randint(1,2)
                if coin == 1:
                    total_utility += (own_value - float(my_bid))
        x_bids.append(my_bid)
        y_utility.append(total_utility/sims)
        if total_utility > best_utility:
            best_utility = total_utility
            best_bid = my_bid
    max_utility, max_bid = max(y_utility), x_bids[np.argmax(y_utility)]
    plt.plot(x_bids, y_utility)
    plt.xlabel ('Bid')
    plt.ylabel ('Utility')
    plt.title ('Utility for Bids')
    plt.savefig('bid_total_{}')
    
    return best_bid, best_utility/sims

for i in range(10, 110, 10):
    print(find_best(i))