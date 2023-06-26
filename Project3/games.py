import numpy as np
import random
import matplotlib.pyplot as plt

def exponential_weights_algorithm_selection(total_payoffs, learning_rate, k_actions, h):
        weights = [(1 + learning_rate)**(total_payoffs[j]/h) for j in range(k_actions)]
        probabilities = [weights[j]/sum(weights) for j in range(k_actions)]
        selection = np.random.choice(k_actions, p=probabilities)
        return selection

def follow_the_perturbed_leader_selection(total_payoffs, noise_level, k_actions):
    max_payoff = max(total_payoffs)
    noise_level = noise_level*max_payoff
    noise = [random.random()*noise_level for i in range(k_actions)]
    options = np.array(noise) + np.array(total_payoffs)
    selection = np.random.choice(np.flatnonzero(options == options.max()))
    return selection

def delayed_exponential_weights_algorithm_selection(total_payoffs, learning_rate, k_actions, h, round_number):
    if round_number < 10:
        return 0
    else:
        return exponential_weights_algorithm_selection(total_payoffs, learning_rate, k_actions, h)
def fixed_selection(selection):
    return selection

def prisoner_dilemma (player1_algo, player2_algo, payoff_matrix, n_games, learning_rate_1=None, learning_rate_2=None):
    player_1_action_stream = []
    player_1_payoff_stream = []
    player_1_payoffs = [0, 0]
    player_2_action_stream = []
    player_2_payoff_stream = []
    player_2_payoffs = [0, 0]
    for i in range(n_games):
        if player1_algo == exponential_weights_algorithm_selection:
            player_1_selection = player1_algo(player_1_payoffs, learning_rate_1, 2, 5)
        elif player1_algo == follow_the_perturbed_leader_selection:
            player_1_selection = player1_algo(player_1_payoffs, learning_rate_1, 2)
        elif player1_algo == fixed_selection:
            player_1_selection = player1_algo(learning_rate_1)
        player_1_action_stream.append(player_1_selection)

        if player2_algo == exponential_weights_algorithm_selection:
            player_2_selection = player2_algo(player_2_payoffs, learning_rate_2, 2, 5)
        elif player2_algo == follow_the_perturbed_leader_selection:
            player_2_selection = player2_algo(player_2_payoffs, learning_rate_2, 2)
        elif player2_algo == fixed_selection:
            player_2_selection = player2_algo(learning_rate_2)
        
        player_2_action_stream.append(player_2_selection)

        current_payoff = payoff_matrix[player_1_selection] [player_2_selection] 

        player_1_payoffs[0] += payoff_matrix[0][player_2_selection][0]
        player_1_payoffs[1] += payoff_matrix[1][player_2_selection][0]

        player_2_payoffs[0] += payoff_matrix[player_1_selection][0][1]
        player_2_payoffs[1] += payoff_matrix[player_1_selection][1][1]
        

        player_1_payoff_stream.append(current_payoff[0])
        player_2_payoff_stream.append(current_payoff[1])
    return player_1_payoff_stream, player_1_action_stream, player_2_payoff_stream, player_2_action_stream

def calculate_fpa_utilities(bid1, player_1_value, bid2, player_2_value, selection_to_bid):
    p1_current_utilities = [0 for i in range(len(selection_to_bid))]
    p2_current_utilities = [0 for i in range(len(selection_to_bid))]
    potential_bid1 = [selection_to_bid[j]*player_1_value for j in range(len(selection_to_bid))]
    potential_bid2 = [selection_to_bid[j]*player_2_value for j in range(len(selection_to_bid))]

    for i in range(len(selection_to_bid)):
        if potential_bid1[i] > bid2:
            p1_current_utilities[i] = player_1_value - potential_bid1[i]
        elif potential_bid1[i] < bid2:
            p1_current_utilities[i] = 0
        else:
            if random.random() > 0.5:
                p1_current_utilities[i] = player_1_value - potential_bid1[i]
            else:
                p1_current_utilities[i] = 0

        if potential_bid2[i] > bid1:
            p2_current_utilities[i] = player_2_value - potential_bid2[i]
        elif potential_bid2[i] < bid1:
            p2_current_utilities[i] = 0
        else:
            if random.random() > 0.5:
                p2_current_utilities[i] = player_2_value - potential_bid2[i]
            else:
                p2_current_utilities[i] = 0
    return p1_current_utilities, p2_current_utilities

def calculate_gspa_utilities(player_1_bid, player_1_value, player_2_bid, player_2_value, bid_choices, prob):
    p1_current_utilities = [0 for i in range(len(bid_choices))]
    p2_current_utilities = [0 for i in range(len(bid_choices))]

    for i in range(len(bid_choices)):
        p1_potential_bid = bid_choices[i] * player_1_value
        
        if p1_potential_bid > player_2_bid:
            p1_current_utilities [i] = prob[0]*(player_1_value - player_2_bid)
        elif p1_potential_bid < player_2_bid:
            p1_current_utilities [i] = prob[1]*(player_1_value)
        else:
            if random.random() > 0.5:
                p1_current_utilities [i] = prob[1]*(player_1_value - player_2_bid)
            else:
                p1_current_utilities [i] = prob[0]*(player_1_value)

        p2_potential_bid = bid_choices[i] * player_2_value
        
        if p2_potential_bid > player_1_bid:
            p2_current_utilities [i] = prob[1]*(player_2_value - player_1_bid)
        elif p2_potential_bid < player_1_bid:
            p2_current_utilities [i] = prob[0]*(player_2_value)
        else:
            if random.random() > 0.5:
                p2_current_utilities [i] = prob[1]*(player_2_value - player_1_bid)
            else:
                p2_current_utilities [i] = prob[0]*(player_2_value)
    return p1_current_utilities, p2_current_utilities

def fpa (player1_algo, player2_algo, n_games, learning_rate_1=None, learning_rate_2=None):
    selection_to_bid = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    player_1_value_stream = []
    player_2_value_stream = []
    player_1_bid_stream = []
    player_1_utility_stream = []
    player_1_utilities = [0 for j in range(len(selection_to_bid))]
    player_2_bid_stream = []
    player_2_utility_stream = []
    player_2_utilities = [0 for j in range(len(selection_to_bid))]
    for i in range(n_games):
        player_1_value = random.random()*10
        player_1_value_stream.append(player_1_value)
        player_2_value = random.random()*10
        player_2_value_stream.append(player_2_value)
        if player1_algo == exponential_weights_algorithm_selection:
            player_1_selection = player1_algo(player_1_utilities, learning_rate_1, 7, 10)
        elif player1_algo == follow_the_perturbed_leader_selection:
            player_1_selection =  player1_algo(player_1_utilities, learning_rate_1, 7)
        elif player1_algo == fixed_selection:
            player_1_selection = player1_algo(learning_rate_1)
        elif delayed_exponential_weights_algorithm_selection:
            player_1_selection = player1_algo(player_1_utilities, learning_rate_1, 7, 10, i)

        if player2_algo == exponential_weights_algorithm_selection:
            player_2_selection = player2_algo(player_2_utilities, learning_rate_2, 7, 10)
        elif player2_algo == follow_the_perturbed_leader_selection:
            player_2_selection = player2_algo(player_2_utilities, learning_rate_2, 7)
        elif player2_algo == fixed_selection:
            player_2_selection = player2_algo(learning_rate_2)
        elif delayed_exponential_weights_algorithm_selection:
            player_2_selection = player2_algo(player_2_utilities, learning_rate_2, 7, 10, i)
        
        bid1 = selection_to_bid[player_1_selection]*player_1_value
        player_1_bid_stream.append(selection_to_bid[player_1_selection])

        bid2 = selection_to_bid[player_2_selection]*player_2_value
        player_2_bid_stream.append(selection_to_bid[player_2_selection])
        
        p1_current_utilities, p2_current_utilities = calculate_fpa_utilities(bid1, player_1_value, bid2, player_2_value, selection_to_bid)
        
        
        player_1_utilities = list(np.array(player_1_utilities) + np.array(p1_current_utilities))
        player_2_utilities = list(np.array(player_2_utilities) + np.array(p2_current_utilities))
        player_1_utility_stream.append(p1_current_utilities[player_1_selection])
        player_2_utility_stream.append(p2_current_utilities[player_2_selection])
    return player_1_utility_stream, player_1_bid_stream, player_1_value_stream, player_2_utility_stream, player_2_bid_stream, player_2_value_stream

def gspa (player1_algo, player2_algo, n_games, learning_rate_1=None, learning_rate_2=None, click_probabilities = [0.25, 0.75]):
    bid_choices = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    player_1_value_stream = []
    player_2_value_stream = []
    player_1_bid_stream = []
    player_1_utility_stream = []
    player_1_utilities = [0 for j in range(len(bid_choices))]
    player_2_bid_stream = []
    player_2_utility_stream = []
    player_2_utilities = [0 for j in range(len(bid_choices))]
    for i in range(n_games):
        player_1_value = random.random()*10
        player_2_value = random.random()*10

        player_1_value_stream.append(player_1_value)
        player_2_value_stream.append(player_2_value)

        if player1_algo == exponential_weights_algorithm_selection:
            player_1_selection = player1_algo(player_1_utilities, learning_rate_1, 11, 10)
        elif player1_algo == follow_the_perturbed_leader_selection:
            player_1_selection = player1_algo(player_1_utilities, learning_rate_1, 11)
        elif player1_algo == fixed_selection:
            player_1_selection = player1_algo(learning_rate_1)

        if player2_algo == exponential_weights_algorithm_selection:
            player_2_selection = player2_algo(player_2_utilities, learning_rate_2, 11, 10)
        elif player2_algo == follow_the_perturbed_leader_selection:
            player_2_selection = player2_algo(player_2_utilities, learning_rate_2, 11)
        elif player2_algo == fixed_selection:
            player_2_selection = player2_algo(learning_rate_2)

        player_1_bid = bid_choices[player_1_selection]*player_1_value
        player_1_bid_stream.append(bid_choices[player_1_selection])

        player_2_bid = bid_choices[player_2_selection]*player_2_value
        player_2_bid_stream.append(bid_choices[player_2_selection])

        p1_current_utilities, p2_current_utilities = calculate_gspa_utilities(player_1_bid, player_1_value, player_2_bid, player_2_value, bid_choices, click_probabilities)

        player_1_utilities = list(np.array(player_1_utilities) + np.array(p1_current_utilities))
        player_2_utilities = list(np.array(player_2_utilities) + np.array(p2_current_utilities))
        player_1_utility_stream.append(p1_current_utilities[player_1_selection])
        player_2_utility_stream.append(p2_current_utilities[player_2_selection])
        
    return player_1_utility_stream, player_1_bid_stream, player_1_value_stream, player_2_utility_stream, player_2_bid_stream, player_2_value_stream

def prisoner_graph():
    player1_algo = exponential_weights_algorithm_selection
    learning_rate_1 = 0.166
    player2_algo = follow_the_perturbed_leader_selection
    learning_rate_2 = 0.25
    n_games = 50
    n_simulations = 100
    games = []
    pd_payoffs = [[(2, 2), (-1, 4)], [(4, -1), (2, 2)]]
    player1_total_payoffs = [0 for i in range(n_games)]
    player1_total_bid = [0 for i in range(n_games)]

    player2_total_payoffs = [0 for i in range(n_games)]
    player2_total_bid = [0 for i in range(n_games)]
    for i in range(n_games):
        games.append(i + 1)
    for j in range(n_simulations):
        player_1_payoff_stream, player_1_action_stream, player_2_payoff_stream, player_2_action_stream = prisoner_dilemma(player1_algo, player2_algo, pd_payoffs, n_games, learning_rate_1, learning_rate_2)

        
        player1_total_payoffs = list(np.array(player1_total_payoffs) + np.array(player_1_payoff_stream))
        player1_total_bid = list(np.array(player1_total_bid) + np.array(player_1_action_stream))

        player2_total_payoffs = list(np.array(player2_total_payoffs) + np.array(player_2_payoff_stream))
        player2_total_bid = list(np.array(player2_total_bid) + np.array(player_2_action_stream))
    
    p1_average_payoff = [player1_total_payoffs[i]/n_simulations for i in range(n_games)]
    p1_average_bid = [player1_total_bid[i]/n_simulations for i in range(n_games)]

    p2_average_payoff = [player2_total_payoffs[i]/n_simulations for i in range(n_games)]
    p2_average_bid = [player2_total_bid[i]/n_simulations for i in range(n_games)]

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(games, p1_average_payoff, label="Exponential Weight")
    ax.plot(games, p2_average_payoff, label="Follow the Perturbed Leader")
    ax.set_title('Payoffs of Exponential Weight vs. Follow the Perturbed Leader in Prisoner Dilemma, close payoffs')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Average Payoff at Game')
    ax.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10,5))
    ax2.plot(games, p1_average_bid, label="Exponential Weight")
    ax2.plot(games, p2_average_bid, label="Follow the Perturbed Leader")
    ax2.set_title('Action of Exponential Weight vs. Follow the Perturbed Leader in Prisoner Dilemma, close payoffs')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Average Action at Game')
    ax2.legend()

    fig.savefig('CP Prisoner Payoff E Vs Pert.png')
    fig2.savefig('CP Prisoner Action E Vs Pert.png')

def fpa_graph():
    player1_algo = follow_the_perturbed_leader_selection
    learning_rate_1 = 0.25
    player2_algo = delayed_exponential_weights_algorithm_selection
    learning_rate_2 = 0.166
    n_games = 50
    n_simulations = 100
    games = []
    player1_total_payoffs = [0 for i in range(n_games)]
    player1_total_bid = [0 for i in range(n_games)]

    player2_total_payoffs = [0 for i in range(n_games)]
    player2_total_bid = [0 for i in range(n_games)]
    for i in range(n_games):
        games.append(i + 1)
    for j in range(n_simulations):
        player_1_payoff_stream, player_1_action_stream, v1, player_2_payoff_stream, player_2_action_stream,v2 = fpa(player1_algo, player2_algo, n_games, learning_rate_1, learning_rate_2)

        
        player1_total_payoffs = list(np.array(player1_total_payoffs) + np.array(player_1_payoff_stream))
        player1_total_bid = list(np.array(player1_total_bid) + np.array(player_1_action_stream))

        player2_total_payoffs = list(np.array(player2_total_payoffs) + np.array(player_2_payoff_stream))
        player2_total_bid = list(np.array(player2_total_bid) + np.array(player_2_action_stream))
    
    p1_average_payoff = [player1_total_payoffs[i]/n_simulations for i in range(n_games)]
    p1_average_bid = [player1_total_bid[i]/n_simulations for i in range(n_games)]

    p2_average_payoff = [player2_total_payoffs[i]/n_simulations for i in range(n_games)]
    p2_average_bid = [player2_total_bid[i]/n_simulations for i in range(n_games)]

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(games, p1_average_payoff, label="FTPL")
    ax.plot(games, p2_average_payoff, label="Devised Algorithm")
    ax.set_title('Payoffs of FTPL vs. Devised Algorithm in FPA')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Average Payoff at Game')
    ax.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10,5))
    ax2.plot(games, p1_average_bid, label="FTPL")
    ax2.plot(games, p2_average_bid, label="Devised Algorithm")
    ax2.set_title('Bid of FTPL vs. Devised Algorithm in FPA')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Average Bid at Game')
    ax2.legend()

    fig.savefig('FPA Devised Payoff.png')
    fig2.savefig('FPA Devised Bid.png')

def gspa_graph():
    
    player1_algo = exponential_weights_algorithm_selection
    learning_rate_1 = 0.166
    player2_algo = follow_the_perturbed_leader_selection
    learning_rate_2 = 0.5
    n_games = 50
    n_simulations = 100
    games = []
    player1_total_payoffs = [0 for i in range(n_games)]
    player1_total_bid = [0 for i in range(n_games)]
    probs = [0.5, 0.5]
    player2_total_payoffs = [0 for i in range(n_games)]
    player2_total_bid = [0 for i in range(n_games)]
    for i in range(n_games):
        games.append(i + 1)
    for j in range(n_simulations):
        player_1_payoff_stream, player_1_action_stream, v1, player_2_payoff_stream, player_2_action_stream,v2 = gspa(player1_algo, player2_algo, n_games, learning_rate_1, learning_rate_2, probs)

        
        player1_total_payoffs = list(np.array(player1_total_payoffs) + np.array(player_1_payoff_stream))
        player1_total_bid = list(np.array(player1_total_bid) + np.array(player_1_action_stream))

        player2_total_payoffs = list(np.array(player2_total_payoffs) + np.array(player_2_payoff_stream))
        player2_total_bid = list(np.array(player2_total_bid) + np.array(player_2_action_stream))
    
    p1_average_payoff = [player1_total_payoffs[i]/n_simulations for i in range(n_games)]
    p1_average_bid = [player1_total_bid[i]/n_simulations for i in range(n_games)]

    p2_average_payoff = [player2_total_payoffs[i]/n_simulations for i in range(n_games)]
    p2_average_bid = [player2_total_bid[i]/n_simulations for i in range(n_games)]

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(games, p1_average_payoff, label="Exponential Weights")
    ax.plot(games, p2_average_payoff, label="FTPL")
    ax.set_title('Payoffs of EW vs. FTPL in GSPA, 50 percent click')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Average Payoff at Game')
    ax.legend()

    fig2, ax2 = plt.subplots(1,1,figsize=(10,5))
    ax2.plot(games, p1_average_bid, label="Exponential Weights")
    ax2.plot(games, p2_average_bid, label="FTPL")
    ax2.set_title('Bid of EW vs. FTPL = value in GSPA, 50 percent click')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Average Bid at Game')
    ax2.legend()

    fig.savefig('GSPA Payoff EW Vs FTPL1half.png')
    fig2.savefig('GSPA Action EW Vs FTPL1half.png')
"""prisoner_graph()"""
fpa_graph()
"""gspa_graph()"""
"""n_games = 50
learning_rate = 0.166
pd_payoffs = [[(3, 3), (-1, 5)], [(5, -1), (1, 1)]]"""
"""u1, b1, v1, u2, b2, v2 = fpa(exponential_weights_algorithm_selection, follow_the_perturbed_leader_selection, n_games, learning_rate, learning_rate)
print("Player 1 utilities")
print(u1)
print("Player 1 bids")
print(b1)
print("Player 1 values")
print(v1)
print("Player 2 utilities")
print(u2)
print("Player 2 bids")
print(b2)
print("Player 2 values")
print(v2)"""

"""p1, a1, p2, a2, = prisoner_dilemma(exponential_weights_algorithm_selection, follow_the_perturbed_leader_selection, pd_payoffs, n_games, learning_rate, learning_rate)
print("Player 1 payoffs")
print(p1)
print("Player 1 actions")
print(a1)

print("Player 2 payoffs")
print(p2)
print("Player2 actions")
print(a2)"""
"""
probabilities = [0.75, 0.25]
u1, b1, v1, u2, b2, v2 = gspa(exponential_weights_algorithm_selection, follow_the_perturbed_leader_selection, n_games, learning_rate, learning_rate, probabilities)
print("Player 1 utilities")
print(u1)
print("Player 1 bids")
print(b1)
print("Player 1 values")
print(v1)
print("Player 2 utilities")
print(u2)
print("Player 2 bids")
print(b2)
print("Player 2 values")
print(v2)"""

