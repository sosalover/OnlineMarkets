import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import sys as sys

def adversarial_data_gen(rounds, actions):
    data = []
    for round in range(rounds):
        data.append([])
        for action in range(actions):
            data[len(data) - 1].append(random.random()*int((round % actions) == action) * (round + 1))
    return data
print(adversarial_data_gen(20, 5))
def tennis_data_generator():
    file = open("Project2/tennis_data.csv")
    payoff_data = np.zeros([10, 32])
    ind = 0
    for line in file:
        data = line.split(",")
        del data[0]
        for data_ind, payoff in enumerate(data):
            payoff_data[ind % 10][int(ind / 10) + (4 * int(data_ind))] = int(payoff)
        ind += 1
    return payoff_data


def adversarial_fair_data_generator(n, k):
    data = []
    total_payoffs = [0 for x in range(k)]
    for i in range(n):
        payoff = random.random()
        action = np.argmin(np.array(total_payoffs))
        action_payoffs = [0 for a in range(k)]
        action_payoffs[action] = payoff
        data.append(action_payoffs)
        total_payoffs[action] += payoff
    return data

"""print(adversarial_fair_data_generator(50, 5))
"""
def bernoulli_data_generator(n, k):
    data = []
    probabilities = [random.random()*1/2 for x in range(k)]
    """print(probabilities)"""
    for i in range(n):
        action_payoffs = [0 for a in range(k)]
        for j in range(k):
            action_payoffs[j] = np.random.choice(2, p=[1-probabilities[j], probabilities[j]])
        data.append(action_payoffs)
    return data
def add_arrays (lst):
    totals = np.array([0 for x in range(len(lst[0]))])
    for i in range(len(lst)):
        totals += np.array(lst[i])
    print(totals)

def exponential_weights_algorithm(data, learning_rate):
    n = len(data)
    k = len(data[0])
    total_payoffs = [0 for x in range(k)]
    h = 100
    winnings = 0
    for i in range(n):
        weights = [(1 + learning_rate)**(total_payoffs[j]/h) for j in range(k)]
        probabilities = [weights[j]/sum(weights) for j in range(k)]
        selection = np.random.choice(k, p=probabilities)
        winnings += data[i][selection]
        total_payoffs = list(np.array(total_payoffs) + np.array(data[i]))
    return winnings, total_payoffs, weights


"""add_arrays(bernoulli_data_generator(100, 5))"""


   


def exponential_weight_adversarial(learning_rate, k, r):
    """generated_data = []"""
    winnings = 0
    total_payoffs = [0 for x in range(k)]
    weights = [1 for x in range(k)]
    action_prob = [1/k for x in range(k)]

    "rounds"
    h = 1
    for i in range (r):
        this_payoff = random.random()
        payoff_action = np.argmin(np.array(total_payoffs))
        current_round_payoffs = [0 for x in range(k)]
        current_round_payoffs[payoff_action] = this_payoff
        """generated_data.append(current_round_payoffs)"""
        for c in range(k):
            weights[c] = (1 + learning_rate)**(total_payoffs[c]/h)
        for c in range(k):
            action_prob[c] = weights[c]/sum(weights)
        selection = np.random.choice(k, p=action_prob)
        if selection == payoff_action:
            winnings += this_payoff
        total_payoffs[payoff_action] += this_payoff
    
    return winnings, total_payoffs, weights

"""w, g, we = exponential_weight_adversarial(0.5, 6, 30)
print("winnings")
print(w)
print("weights")
print(we)
print("generated data")
print(g)"""
"""
Fix a probability for each action p1,...,pk with each pk in [0,1/2].
In each round i, draw the payoff of each action j as vji ~ B(pj) 
(i.e, from the Bernoulli distribution Links to an external site.
with probability pj of being 1 and probability 1-pj of being 0).
"""


def exponential_weight_bernoulli(learning_rate, k, r):
    payoff_probabilities = [random.random()*1/2 for x in range (k)]
    """print(payoff_probabilities)"""
    """generated_data = []"""
    winnings = 0
    total_payoffs = [0 for x in range(k)]
    weights = [1 for x in range(k)]
    action_prob = [1/k for x in range(k)]

    "rounds"
    h = 7
    for i in range (r):
        current_round_payoffs = [0 for x in range(k)]
        for a in range(k):
            current_round_payoffs[a] = np.random.choice(2, p=[1-payoff_probabilities[a], payoff_probabilities[a]])
        """generated_data.append(current_round_payoffs)"""
        for c in range(k):
            weights[c] = (1 + learning_rate)**(total_payoffs[c]/h)
        for c in range(k):
            action_prob[c] = weights[c]/sum(weights)
        selection = np.random.choice(k, p=action_prob)
        winnings += current_round_payoffs[selection]
        total_payoffs = list(np.array(total_payoffs) + np.array(current_round_payoffs))
    
    return winnings, total_payoffs, weights

def monte_carlo_sim(l, data, simulation_rounds):
    total_function_winnings = 0
    total_hindsight_winnings = 0
    for i in range(simulation_rounds):
        winnings, total_payoffs, weights = exponential_weights_algorithm(data, l)
        total_function_winnings += winnings
        hindsight_winnings = max(total_payoffs)
        total_hindsight_winnings += hindsight_winnings
    avg_function_winnings = total_function_winnings/simulation_rounds
    avg_hindsight_winnings = total_hindsight_winnings/simulation_rounds

    return avg_function_winnings, avg_hindsight_winnings, weights

def graph():
    k = 6
    n = 100
    
    monte_carlo_simulations = 500
    function_payoffs = []
    hindsight_payoffs = []
    learning_rates = [0.001, 0.1, 0.2, (np.log(k)/n)**(1/2), 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 8, 1000]
    learning_rates = list(np.sort(np.array(learning_rates)))
    print(learning_rates)
    
    for i in range(len(learning_rates)):
        data = adversarial_data_gen(n, k    )
        avg_function_winnings, avg_hindsight_winnings, weights = monte_carlo_sim(learning_rates[i], data, monte_carlo_simulations)
        function_payoffs.append(avg_function_winnings)
        hindsight_payoffs.append(avg_hindsight_winnings)
    for i in range (len(learning_rates)):
        learning_rates[i] = np.log(learning_rates[i])
    print(learning_rates)
    regret = [(hindsight_payoffs[q] - function_payoffs[q])/n for q in range(len(hindsight_payoffs))]
    print(regret)
    
    plt.plot(learning_rates, regret)
    plt.xlabel("Learning Rates (log)")
    plt.ylabel("Average Regret")
    plt.show()


"""w, g, we = exponential_weight_bernoulli(0.5, 6, 30)
print("winnings")
print(w)
print("weights")
print(we)
print("generated data")
print(g)"""

graph()