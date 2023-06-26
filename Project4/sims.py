import numpy as np
import random
import matplotlib.pyplot as plt
import copy

def exponential_weights_algorithm_selection(total_payoffs, learning_rate, k_actions, h):
    weights = [(1 + learning_rate)**(total_payoffs[j]/h) for j in range(k_actions)]
    probabilities = [weights[j]/sum(weights) for j in range(k_actions)]
    selection = np.random.choice(k_actions, p=probabilities)
    return selection

def calculate_spa_reserve_revenue(reserve_price, bidder_values, k_items):
    sorted_values = sorted(copy.deepcopy(bidder_values), reverse=True)
    revenue = 0
    for i in range(k_items):
        if sorted_values[i] > reserve_price:
            revenue += max(reserve_price, sorted_values[k_items])
    return revenue


def spa_reserve(n_rounds, n_bidders, k_items, learning_rate):
    reserve_choices = [x*0.1 for x in range (20)]
    total_revenue = 0
    revenue_stream = []
    reserve_choice_stream = []
    total_reserve_payoffs = [0 for x in range (len(reserve_choices))]
    bidder_value_stream = [[] for _ in range(n_rounds)]
    for i in range(n_rounds):
        bidder_values = [[] for _ in range(n_bidders)]
        for j in range(n_bidders):
            bidder_values[j] = (np.random.exponential(1/2))
        bidder_value_stream[i]=bidder_values
        choice = exponential_weights_algorithm_selection(total_reserve_payoffs, learning_rate, len(reserve_choices), 1)
        this_round_reserve = reserve_choices[choice]
        reserve_choice_stream.append(this_round_reserve)
        this_round_revenue = calculate_spa_reserve_revenue(this_round_reserve, bidder_values, k_items)
        total_revenue += this_round_revenue
        revenue_stream.append(this_round_revenue)
        for j in range(len(total_reserve_payoffs)):
            total_reserve_payoffs[j] += calculate_spa_reserve_revenue(reserve_choices[j], bidder_values, k_items)

    
    return reserve_choice_stream, bidder_value_stream, revenue_stream

def selling_intro_exponential_weights(payoff_matrix, learning_rate, h):
    n_employers = len(payoff_matrix)
    n_employees = len(payoff_matrix[0])
    beliefs = [[0 for i in range(n_employees)] for j in range(n_employers)]
    for i in range(n_employers):
        for j in range(n_employees):
            belief = exponential_weights_algorithm_selection(payoff_matrix[i][j], learning_rate, len(payoff_matrix[i][j]), 10)
            beliefs[i][j] = belief
    return beliefs

n_employees = 2
n_employers = 3
employee_distributions_G = [[random.randint(0,9) for i in range(n_employees)] for b in range(n_employers)]
employer_distributions_G = [[random.randint(0,9) for i in range(n_employees)] for b in range(n_employers)]

def selling_introductions(n_employees, n_employers, n_rounds, learning_rate):
    belief_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    employee_payoff_matrix = [[[0 for c in range(10)] for a in range(n_employees)] for b in range(n_employers)]
    employer_payoff_matrix = [[[0 for c in range(10)] for a in range(n_employees)] for b in range(n_employers)]
    employee_distributions = employee_distributions_G
    employer_distributions = employer_distributions_G
    employer_value_stream = []
    employee_value_stream = []
    optimal_revenue_stream = []
    choice_stream = []
    revenue_stream = []
    optimal_choice_stream = []
    #access via distributions[employer][employee]
    for i in range(n_rounds):
        this_round_employer_beliefs_i = selling_intro_exponential_weights(employer_payoff_matrix, learning_rate, 9)
        this_round_employee_beliefs_i = selling_intro_exponential_weights(employee_payoff_matrix, learning_rate, 9)


        employee_values = [[random.uniform(0, employee_distributions[y][x]) for x in range(n_employees)] for y in range(n_employers)]
        employer_values = [[random.uniform(0, employer_distributions[y][x]) for x in range(n_employees)] for y in range(n_employers)]

        employee_value_stream.append(employee_values)
        employer_value_stream.append(employer_values)

        
        this_round_employer_payoffs = [[[0 for c in range(10)] for a in range(n_employees)] for b in range(n_employers)]
        this_round_employee_payoffs = [[[0 for c in range(10)] for a in range(n_employees)] for b in range(n_employers)]

        this_round_choices = [[0 for a in range(n_employees)] for b in range(n_employers)]

        for er_belief_i in range(len(belief_map)):
            for ee_belief_i in range(len(belief_map)):
                er_belief = belief_map[er_belief_i]
                ee_belief = belief_map[ee_belief_i]
                for er_i in range(n_employers):
                    for ee_i in range(n_employees):
                        er_ei_payout, this_round_choices[er_i][ee_i] = calculate_selling_introductions(employer_values, employee_values, er_i, ee_i, er_belief, ee_belief)
                        this_round_employer_payoffs[er_i][ee_i][er_belief_i], this_round_employee_payoffs[er_i][ee_i][ee_belief_i] = er_ei_payout, er_ei_payout
        
        employer_payoff_matrix = (np.add(np.array(employer_payoff_matrix), np.array(this_round_employer_payoffs))).tolist()
        employee_payoff_matrix = (np.add(np.array(employee_payoff_matrix), np.array(this_round_employee_payoffs))).tolist()


        this_round_revenue = 0
        
        for er_i in range(n_employers):
            for ee_i in range(n_employees):
                this_round_revenue += this_round_employer_payoffs[er_i][ee_i][this_round_employer_beliefs_i[er_i][ee_i]] + this_round_employee_payoffs[er_i][ee_i][this_round_employee_beliefs_i[er_i][ee_i]]

        optimal_revenue = 0
        optimal_choice = [[0 for i in range(n_employees)] for b in range(n_employers)]
        for er_i in range(n_employers):
            for ee_i in range(n_employees):
                rev_temp, optimal_choice[er_i][ee_i] = calculate_selling_introductions(employer_values, employee_values, er_i, ee_i, employer_distributions[er_i][ee_i], employee_distributions[er_i][ee_i])
                optimal_revenue += rev_temp        
        
        optimal_revenue_stream.append(optimal_revenue)
        optimal_choice_stream.append(optimal_choice)
        choice_stream.append(this_round_choices)
        revenue_stream.append(this_round_revenue)
    
    return employer_distributions, employee_distributions, choice_stream, revenue_stream, optimal_choice_stream, optimal_revenue_stream

def calculate_selling_introductions(employer_values, employee_values, er_i, ee_i, er_belief, ee_belief):
    
    employer_v = employer_values[er_i][ee_i]
    employee_v = employee_values[er_i][ee_i]
    employer_virtual_value = 2*employer_v - er_belief
    employee_virtual_value = 2*employee_v - ee_belief

    if employer_virtual_value + employee_virtual_value < 0:
        revenue = 0
        choice = 0
    else:
        min_employer_price = (ee_belief + er_belief - 2*employee_v)/2
        min_employee_price = (ee_belief + er_belief - 2*employer_v)/2
        revenue = min_employee_price + min_employer_price
        choice = 1
    return revenue, choice


def monte_carlo(n_simulations, game, args):
    n_rounds = args[0]
    total_roundly_reserve_choices = [0 for _ in range (n_rounds)]
    total_roundly_revenue = [0 for _ in range(n_rounds)]
    for i in range(n_simulations):
        n_rounds, n_bidders, k_items, learning_rate = args[0], args[1], args[2], args[3]
        if game == spa_reserve:
            
            this_sim_reserve_choice_stream, bidder_value_stream, this_sim_revenue_stream = spa_reserve(n_rounds, n_bidders, k_items, learning_rate)
            total_roundly_reserve_choices = list(np.array(total_roundly_reserve_choices) + np.array(this_sim_reserve_choice_stream))
            total_roundly_revenue = list(np.array(total_roundly_revenue) + np.array(this_sim_revenue_stream))
        
    avg_roundly_reserve_choices = list(np.array(total_roundly_reserve_choices)/n_simulations)
    avg_roundly_revenue = list(np.array(total_roundly_revenue)/n_simulations)

    return avg_roundly_reserve_choices, avg_roundly_revenue

def monte_carlo_SI(n_simulations, args):
        n_employees, n_employers, n_rounds, learning_rate = args[0], args[1], args[2], args[3]
        total_roundly_revenue = [0 for _ in range(n_rounds)]
        total_optimal_revenue = [0 for _ in range(n_rounds)]
        total_roundly_choice = [[[0 for i in range(n_employees)] for b in range(n_employers)] for _ in range(n_rounds)]
        total_optimal_choice = [[[0 for i in range(n_employees)] for b in range(n_employers)] for _ in range(n_rounds)]
        for i in range(n_simulations):
            print(i)
            employer_distributions, employee_distributions, choice_stream, revenue_stream, optimal_choice_stream, optimal_revenue_stream = selling_introductions(n_employees, n_employers, n_rounds, learning_rate)
            total_roundly_revenue = list(np.array(total_roundly_revenue) + np.array(revenue_stream))
            total_optimal_revenue = list(np.array(total_optimal_revenue) + np.array(optimal_revenue_stream))
        avg_roundly_revenue = list(np.array(total_roundly_revenue)/n_simulations)
        avg_optimal_revenue = list(np.array(total_optimal_revenue)/n_simulations)

        return avg_roundly_revenue, avg_optimal_revenue


def graph_si(n_simulations, args):
    n_employees, n_employers, n_rounds, learning_rate = args[0], args[1], args[2], args[3]
    rounds = [h for h in range(n_rounds)]
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    avg_roundly_revenue, avg_optimal_revenue = monte_carlo_SI(n_simulations, args)
    ax.plot(rounds, avg_roundly_revenue, label = "Exponential Weights")
    ax.plot(rounds, avg_optimal_revenue, label = "Optimal")
    ax.set_title('Optimal vs Exponential Weights Revenue for Selling Introductions')
    ax.set_xlabel('Round')
    ax.set_ylabel('Revenue')
    ax.legend()
    

    plt.show()

def graph_spa_reserve(n_simulations, args):
    n_rounds, n_bidders, k_items, learning_rate = args[0], args[1], args[2], args[3]
    rounds = [h for h in range(n_rounds)]
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    avg_roundly_reserve_choices, avg_roundly_revenue = monte_carlo(n_simulations, spa_reserve, args)
    ax.plot(rounds, avg_roundly_reserve_choices)
    ax.set_title('Average Reserve Choice by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Reserve')
    

    fig2, ax2 = plt.subplots(1,1,figsize=(10,5))
    ax2.plot(rounds, avg_roundly_revenue)
    ax2.set_title('Average Revenue by Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Revenue')

    plt.show()

"""graph_spa_reserve(1000, [100, 2, 1, 1.5])"""
graph_si(1000, [2, 3, 200, 1])
"""employer_distributions, employee_distributions, choice_stream, revenue_stream, optimal_choice_stream, optimal_revenue_stream = selling_introductions(2, 3, 100, 0.1)"""





