import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

random.seed(10)

def exponential_weights_algorithm_selection(total_payoffs, learning_rate, h):
    k_actions = len(total_payoffs)
    weights = [(1 + learning_rate)**(total_payoffs[j]/h) for j in range(k_actions)]
    probabilities = [weights[j]/sum(weights) for j in range(k_actions)]
    selection = np.random.choice(k_actions, p=probabilities)
    return selection

def da_inference(n_bidders, n_rounds, endowed_values):
    discrete_values = [x*.1 for x in range(10)]
    discrete_bids = [x*.1 for x in range(10)]
    learning_rate = math.sqrt(math.log(len(discrete_bids))/n_rounds)
    payoffs = [[0 for i in range(len(discrete_bids))] for j in range(n_bidders)]
    rationalizable_grids = [[0 for i in range(len(discrete_values))] for j in range(n_bidders)]
    qualities_per_round = [None for j in range(n_rounds)]
    bids_per_round = [None for j in range(n_rounds)]
    epsilon_stream = []
    rationalizable_set_stream = [[[] for i in range(n_bidders)] for j in range(n_rounds)]
    point_prediction_stream = [[0 for i in range(n_bidders)] for j in range(n_rounds)]

    for this_round in range(n_rounds):
        qualities = [random.random() for i in range(n_bidders)]
        bids = [None for i in range(n_bidders)]
        qualities_per_round[this_round] = copy.deepcopy(qualities)
        this_round_epsilon = 2* math.sqrt(math.log(len(discrete_bids))/(this_round+1))
        epsilon_stream.append(this_round_epsilon)
        
        #make selections and update payoffs
        for bidder in range(n_bidders):
            bids[bidder] = discrete_bids[exponential_weights_algorithm_selection(payoffs[bidder], learning_rate, 1)]
        for bidder in range(n_bidders):
            update_payoffs(payoffs, qualities, bids, endowed_values, bidder, discrete_bids)
        bids_per_round[this_round] = copy.deepcopy(bids)
        
        
        #calculate rationalizable set for each bidder
        for bidder in range(n_bidders):    
            this_bidder_grid = rationalizable_grids[bidder]
            min_regret = None
            point_prediction = None
            for v in range(len(discrete_values)):
                regret = calculate_regret(this_round, bidder, qualities_per_round, bids_per_round, discrete_values[v], discrete_bids)
                if min_regret == None or regret < min_regret:
                    min_regret = regret
                    point_prediction = discrete_values[v]
                this_bidder_grid[v] = regret <= this_round_epsilon
                if regret <= this_round_epsilon:
                    rationalizable_set_stream[this_round][bidder].append(discrete_values[v])
            point_prediction_stream[this_round][bidder] = point_prediction
        
    return endowed_values, rationalizable_set_stream, point_prediction_stream, epsilon_stream

def update_payoffs(payoffs, qualities, bids, endowed_values, bidder, discrete_bids):
    this_round_payoffs = [0 for i in range(len(discrete_bids))]
    for bid in range(len(discrete_bids)):
        new_bids = copy.deepcopy(bids)
        new_bids[bidder] = discrete_bids[bid]
        this_round_payoffs[bid] = calculate_round_payoff(endowed_values[bidder], new_bids, bidder, qualities)
    payoffs[bidder] = list(np.array(payoffs[bidder]) + np.array(this_round_payoffs))

def calculate_round_payoff(v, bids, bidder, qualities):
    quality_weighted_bids = [bids[i]*qualities[i] for i in range(len(bids))]
    winner = np.argmax(quality_weighted_bids)
    if winner != bidder:
        return 0
    else:
        return v-bids[bidder]

def calculate_regret(round_i, bidder, qualities_per_round, bids_per_round, value, discrete_bids):
    total_payoffs = 0
    opt_total_payoffs = 0
    value_total_payoffs = 0
    for i in range(round_i):
        bids = copy.deepcopy(bids_per_round[i])
        qualities = copy.deepcopy(qualities_per_round[i])
        total_payoffs += calculate_round_payoff(value, bids, bidder, qualities)

    for b in range(len(discrete_bids)):
        value_total_payoffs = 0
        for i in range(round_i):
            bids = copy.deepcopy(bids_per_round[i])
            bids[bidder] = discrete_bids[b]
            qualities = copy.deepcopy(qualities_per_round[i])
            value_total_payoffs += calculate_round_payoff(value, bids, bidder, qualities)
        if value_total_payoffs > opt_total_payoffs:
            opt_total_payoffs = value_total_payoffs

    return (opt_total_payoffs - total_payoffs)/(round_i + 1)       



def da_inference_multi_unit_reserve(n_bidders, n_rounds, endowed_values, k_items):
    discrete_values = [x*.1 for x in range(10)]
    discrete_bids = [x*.1 for x in range(10)]
    discrete_reserves = [x*.1 for x in range(10)]
    learning_rate = math.sqrt(math.log(len(discrete_bids))/n_rounds)
    payoffs = [[0 for i in range(len(discrete_bids))] for j in range(n_bidders)]
    rationalizable_grids = [[0 for i in range(len(discrete_values))] for j in range(n_bidders)]
    qualities_per_round = [None for j in range(n_rounds)]
    bids_per_round = [None for j in range(n_rounds)]
    epsilon_stream = []
    rationalizable_set_stream = [[[] for i in range(n_bidders)] for j in range(n_rounds)]
    point_prediction_stream = [[0 for i in range(n_bidders)] for j in range(n_rounds)]
    reserve_price_stream = [0]
    reserve_price = 0
    revenue_stream = []
    for this_round in range(n_rounds):
        qualities = [random.random() for i in range(n_bidders)]
        bids = [None for i in range(n_bidders)]
        qualities_per_round[this_round] = copy.deepcopy(qualities)
        this_round_epsilon = 2* math.sqrt(math.log(len(discrete_bids))/(this_round+1))
        epsilon_stream.append(this_round_epsilon)
        
        #make selections and update payoffs
        for bidder in range(n_bidders):
            bids[bidder] = discrete_bids[exponential_weights_algorithm_selection(payoffs[bidder], learning_rate, 1)]
        for bidder in range(n_bidders):
            update_payoffs_multi_unit_reserve(payoffs, qualities, bids, endowed_values, bidder, discrete_bids, reserve_price, k_items)
        bids_per_round[this_round] = copy.deepcopy(bids)
        this_round_revenue = calculate_round_revenue(bids, qualities, reserve_price, k_items)
        revenue_stream.append(this_round_revenue)
        #calculate rationalizable set for each bidder
        for bidder in range(n_bidders):    
            this_bidder_grid = rationalizable_grids[bidder]
            min_regret = None
            point_prediction = None
            for v in range(len(discrete_values)):
                regret = calculate_regret_multi(this_round, bidder, qualities_per_round, bids_per_round, discrete_values[v], discrete_bids, reserve_price_stream, k_items)
                if min_regret == None or regret < min_regret:
                    min_regret = regret
                    point_prediction = discrete_values[v]
                this_bidder_grid[v] = regret <= this_round_epsilon
                if regret <= this_round_epsilon:
                    rationalizable_set_stream[this_round][bidder].append(discrete_values[v])
            point_prediction_stream[this_round][bidder] = point_prediction
        this_round_inferred_values = []
        for i in range(n_bidders):
            this_round_inferred_values.append(point_prediction_stream[this_round][i])    
        reserve_price = discrete_reserves[determine_optimal_reserve_price(this_round_inferred_values, payoffs, discrete_reserves, qualities, k_items, discrete_bids)]
        reserve_price_stream.append(reserve_price)
        
    return reserve_price_stream, revenue_stream

def update_payoffs_multi_unit_reserve(payoffs, qualities, bids, endowed_values, bidder, discrete_bids, reserve_price, items):
    this_round_payoffs = [0 for i in range(len(discrete_bids))]
    for bid in range(len(discrete_bids)):
        new_bids = copy.deepcopy(bids)
        new_bids[bidder] = discrete_bids[bid]
        this_round_payoffs[bid] = calculate_round_payoff_multi(endowed_values[bidder], new_bids, bidder, qualities, reserve_price, items)
    payoffs[bidder] = list(np.array(payoffs[bidder]) + np.array(this_round_payoffs))

def calculate_round_payoff_multi(v, bids, bidder, qualities, reserve_price, items):
    quality_weighted_bids = [bids[i]*qualities[i] for i in range(len(bids))]
    winners = get_top_n_indices(quality_weighted_bids, items)
    if bidder not in winners or bids[bidder] < reserve_price:
        return 0
    else:
        return v-bids[bidder]

def calculate_regret_multi(round_i, bidder, qualities_per_round, bids_per_round, value, discrete_bids, reserve_price_stream, items):
    total_payoffs = 0
    opt_total_payoffs = 0
    value_total_payoffs = 0
    for i in range(round_i):
        bids = copy.deepcopy(bids_per_round[i])
        qualities = copy.deepcopy(qualities_per_round[i])
        total_payoffs += calculate_round_payoff_multi(value, bids, bidder, qualities, reserve_price_stream[i], items)

    for b in range(len(discrete_bids)):
        value_total_payoffs = 0
        for i in range(round_i):
            bids = copy.deepcopy(bids_per_round[i])
            bids[bidder] = discrete_bids[b]
            qualities = copy.deepcopy(qualities_per_round[i])
            value_total_payoffs += calculate_round_payoff(value, bids, bidder, qualities)
        if value_total_payoffs > opt_total_payoffs:
            opt_total_payoffs = value_total_payoffs

    return (opt_total_payoffs - total_payoffs)/(round_i + 1)       

def calculate_round_revenue(bids, qualities, reserve_price, items):
    quality_weighted_bids = [bids[i]*qualities[i] for i in range(len(bids))]
    winners = get_top_n_indices(quality_weighted_bids, items)
    total_revenue = 0
    for winner in winners:
        if bids[winner] > reserve_price:
            total_revenue += bids[winner]
        
    return total_revenue

def get_top_n_indices(arr, n):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    return [sorted_arr[i][0] for i in range(n)]

def determine_optimal_reserve_price(this_round_inferred_values, payoffs, discrete_reserves, qualities, items, discrete_bids):
    n_bidders = len(payoffs)
    payoffs = copy.deepcopy(payoffs)
    revenue_totals = [0 for i in range(len(discrete_reserves))]
    for i in range(10):
        bids = []
        for bidder in range(n_bidders):
            bids.append(discrete_bids[exponential_weights_algorithm_selection(payoffs[bidder], .1, 1)])
            update_payoffs(payoffs, qualities, bids, this_round_inferred_values, bidder, discrete_bids)
        for r in range(len(discrete_reserves)):
            revenue_totals[r] += calculate_round_revenue(bids, qualities, discrete_reserves[r], items)
    
    return np.argmax(np.array(revenue_totals))

def da_inference_multi_unit_reserve_fixed_reserve(n_bidders, n_rounds, endowed_values, k_items, reserve_price):
    discrete_values = [x*.1 for x in range(10)]
    discrete_bids = [x*.1 for x in range(10)]
    reserve_price_stream = []
    discrete_reserves = [x*.1 for x in range(10)]
    learning_rate = math.sqrt(math.log(len(discrete_bids))/n_rounds)
    payoffs = [[0 for i in range(len(discrete_bids))] for j in range(n_bidders)]
    rationalizable_grids = [[0 for i in range(len(discrete_values))] for j in range(n_bidders)]
    qualities_per_round = [None for j in range(n_rounds)]
    bids_per_round = [None for j in range(n_rounds)]
    epsilon_stream = []
    rationalizable_set_stream = [[[] for i in range(n_bidders)] for j in range(n_rounds)]
    point_prediction_stream = [[0 for i in range(n_bidders)] for j in range(n_rounds)]
    revenue_stream = []
    for this_round in range(n_rounds):
        qualities = [random.random() for i in range(n_bidders)]
        bids = [None for i in range(n_bidders)]
        qualities_per_round[this_round] = copy.deepcopy(qualities)
        this_round_epsilon = 2* math.sqrt(math.log(len(discrete_bids))/(this_round+1))
        epsilon_stream.append(this_round_epsilon)
        
        #make selections and update payoffs
        for bidder in range(n_bidders):
            bids[bidder] = discrete_bids[exponential_weights_algorithm_selection(payoffs[bidder], learning_rate, 1)]
        for bidder in range(n_bidders):
            update_payoffs_multi_unit_reserve(payoffs, qualities, bids, endowed_values, bidder, discrete_bids, reserve_price, k_items)
        bids_per_round[this_round] = copy.deepcopy(bids)
        this_round_revenue = calculate_round_revenue(bids, qualities, reserve_price, k_items)
        revenue_stream.append(this_round_revenue)
        #calculate rationalizable set for each bidder
        for bidder in range(n_bidders):    
            this_bidder_grid = rationalizable_grids[bidder]
            min_regret = None
            point_prediction = None
            for v in range(len(discrete_values)):
                regret = calculate_regret_multi(this_round, bidder, qualities_per_round, bids_per_round, discrete_values[v], discrete_bids, reserve_price_stream, k_items)
                if min_regret == None or regret < min_regret:
                    min_regret = regret
                    point_prediction = discrete_values[v]
                this_bidder_grid[v] = regret <= this_round_epsilon
                if regret <= this_round_epsilon:
                    rationalizable_set_stream[this_round][bidder].append(discrete_values[v])
            point_prediction_stream[this_round][bidder] = point_prediction
        this_round_inferred_values = []
        for i in range(n_bidders):
            this_round_inferred_values.append(point_prediction_stream[this_round][i])    
        reserve_price_stream.append(reserve_price)
        
    return reserve_price_stream, revenue_stream

def da_inference_multi_unit_optimal_reserve(n_bidders, n_rounds, endowed_values, k_items):
    discrete_values = [x*.1 for x in range(10)]
    discrete_bids = [x*.1 for x in range(10)]
    discrete_reserves = [x*.1 for x in range(10)]
    learning_rate = math.sqrt(math.log(len(discrete_bids))/n_rounds)
    payoffs = [[0 for i in range(len(discrete_bids))] for j in range(n_bidders)]
    rationalizable_grids = [[0 for i in range(len(discrete_values))] for j in range(n_bidders)]
    qualities_per_round = [None for j in range(n_rounds)]
    bids_per_round = [None for j in range(n_rounds)]
    epsilon_stream = []
    rationalizable_set_stream = [[[] for i in range(n_bidders)] for j in range(n_rounds)]
    point_prediction_stream = [[0 for i in range(n_bidders)] for j in range(n_rounds)]
    reserve_price_stream = [0]
    reserve_price = 0
    revenue_stream = []
    for this_round in range(n_rounds):
        qualities = [random.random() for i in range(n_bidders)]
        bids = [None for i in range(n_bidders)]
        qualities_per_round[this_round] = copy.deepcopy(qualities)
        this_round_epsilon = 2* math.sqrt(math.log(len(discrete_bids))/(this_round+1))
        epsilon_stream.append(this_round_epsilon)
        
        #make selections and update payoffs
        for bidder in range(n_bidders):
            bids[bidder] = discrete_bids[exponential_weights_algorithm_selection(payoffs[bidder], learning_rate, 1)]
        for bidder in range(n_bidders):
            update_payoffs_multi_unit_reserve(payoffs, qualities, bids, endowed_values, bidder, discrete_bids, reserve_price, k_items)
        bids_per_round[this_round] = copy.deepcopy(bids)
        this_round_revenue = calculate_round_revenue(bids, qualities, reserve_price, k_items)
        revenue_stream.append(this_round_revenue)
        #calculate rationalizable set for each bidder
        for bidder in range(n_bidders):    
            this_bidder_grid = rationalizable_grids[bidder]
            min_regret = None
            point_prediction = None
            for v in range(len(discrete_values)):
                regret = calculate_regret_multi(this_round, bidder, qualities_per_round, bids_per_round, discrete_values[v], discrete_bids, reserve_price_stream, k_items)
                if min_regret == None or regret < min_regret:
                    min_regret = regret
                    point_prediction = discrete_values[v]
                this_bidder_grid[v] = regret <= this_round_epsilon
                if regret <= this_round_epsilon:
                    rationalizable_set_stream[this_round][bidder].append(discrete_values[v])
            point_prediction_stream[this_round][bidder] = point_prediction
        this_round_inferred_values = endowed_values
        reserve_price = discrete_reserves[determine_optimal_reserve_price(this_round_inferred_values, payoffs, discrete_reserves, qualities, k_items, discrete_bids)]
        reserve_price_stream.append(reserve_price)
        
    return reserve_price_stream, revenue_stream

avg_pp_b1 = []
avg_pp_b2 = []

mse_b1 = []
mse_b2 = []

avg_size_rset_b1 = []
avg_size_rset_b2 = []

avg_rset_val_b1 = []
avg_rset_val_b2 = []

"""num_rounds = 25

for sim in range(num_rounds):
    ev, rss, pps, eps_s = da_inference(2, 100, [0.3, 0.7])
    for round in range(100):
        if len(avg_pp_b1) == round:
            avg_pp_b1.append(0)
            avg_pp_b2.append(0)
            mse_b1.append(0)
            mse_b2.append(0)
            avg_size_rset_b1.append(0)
            avg_size_rset_b2.append(0)
            avg_rset_val_b1.append(0)
            avg_rset_val_b2.append(0)
        
        avg_pp_b1[round] += pps[round][0]
        avg_pp_b2[round] += pps[round][1]
        mse_b1[round] += (ev[0] - pps[round][0]) ** 2
        mse_b2[round] += (ev[1] - pps[round][1]) ** 2
        avg_size_rset_b1[round] += len(rss[round][0])
        avg_size_rset_b2[round] += len(rss[round][1])
        avg_rset_val_b1[round] += sum(rss[round][0]) / len(rss[round][0])
        avg_rset_val_b2[round] += sum(rss[round][1]) / len(rss[round][1])

avg_pp_b1 = [(avg_pp_b1[i] / num_rounds) for i in range(100)]
avg_pp_b2 = [(avg_pp_b2[i] / num_rounds) for i in range(100)]
mse_b1 = [(mse_b1[i] / num_rounds) for i in range(100)]
mse_b2 = [(mse_b2[i] / num_rounds) for i in range(100)]
avg_size_rset_b1 = [(avg_size_rset_b1[i] / num_rounds) for i in range(100)]
avg_size_rset_b2 = [(avg_size_rset_b2[i] / num_rounds) for i in range(100)]
avg_rset_val_b1 = [(avg_rset_val_b1[i] / num_rounds) for i in range(100)]
avg_rset_val_b2 = [(avg_rset_val_b2[i] / num_rounds) for i in range(100)]

pp_x = [[i+1 for i in range(100)], [i+1 for i in range(100)]]
pp_y = [avg_pp_b1, avg_pp_b2]
mse_x = [[i+1 for i in range(100)], [i+1 for i in range(100)]]
mse_y = [mse_b1, mse_b2]
rset_size_x = [[i+1 for i in range(100)], [i+1 for i in range(100)]] 
rset_size_y = [avg_size_rset_b1, avg_size_rset_b2]
rset_val_x = [[i+1 for i in range(100)], [i+1 for i in range(100)]]
rset_val_y = [avg_rset_val_b1, avg_rset_val_b2]

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(pp_x[0], pp_y[0], label="Bidder 1")
ax[0, 0].plot(pp_x[1], pp_y[1], label="Bidder 2")
ax[0, 0].set_title("Point prediction average per round")
ax[0, 0].legend()
ax[0, 1].plot(mse_x[0], mse_y[0], label="Bidder 1")
ax[0, 1].plot(mse_x[1], mse_y[1], label="Bidder 2")
ax[0, 1].set_title("Mean squared error per round")
ax[0, 1].legend()
ax[1, 0].plot(rset_size_x[0], rset_size_y[0], label="Bidder 1")
ax[1, 0].plot(rset_size_x[1], rset_size_y[1], label="Bidder 2")
ax[1, 0].set_title("Rationalizable set size per round")
ax[1, 0].legend()
ax[1, 1].plot(rset_val_x[0], rset_val_y[0], label="Bidder 1")
ax[1, 1].plot(rset_val_x[1], rset_val_y[1], label="Bidder 2")
ax[1, 1].set_title("Rationalizable set value per round")
ax[1, 1].legend()
plt.show()
"""
num_rounds = 2

avg_reserve_price = [0 for i in range(100)]
avg_revenue_learned = [0 for i in range(100)]

for i in range(num_rounds):
    reserve_price_stream, revenue_stream = da_inference_multi_unit_reserve(3, 100, [0.1, 0.3,0.7], 2)
    for j in range(100):
        avg_reserve_price[j] += reserve_price_stream[j]
        avg_revenue_learned[j] += revenue_stream[j]

avg_reserve_price = [cur/num_rounds for cur in avg_reserve_price]
avg_revenue_learned = [cur/num_rounds for cur in avg_revenue_learned]

print("Done 1")

avg_revenue_online = [0 for i in range(100)]

for i in range(num_rounds):
    reserve_price_stream, revenue_stream = da_inference_multi_unit_reserve_fixed_reserve(3, 100, [0.1, 0.3, 0.7], 2, 0.1)
    for j in range(100):
        avg_revenue_online[j] += revenue_stream[j]

avg_revenue_online = [cur/num_rounds for cur in avg_revenue_online]

print("Done 2")

avg_revenue_opt = [0 for i in range(100)]

for i in range(num_rounds):
    reserve_price_stream, revenue_stream = da_inference_multi_unit_optimal_reserve(3, 100, [0.1, 0.3, 0.7], 2)
    for j in range(100):
        avg_revenue_opt[j] += revenue_stream[j]

avg_revenue_opt = [cur/num_rounds for cur in avg_revenue_opt]

print("Done 3")

fig, ax = plt.subplots(2)

fig.suptitle("Part II Graphs")

x = [i + 1 for i in range(100)]


ax[0].set_title("Rounds vs Revenue")
ax[0].plot(x, avg_revenue_learned, label="learned reserve price")
ax[0].plot(x, avg_revenue_online, label="Online max") 
ax[0].plot(x, avg_revenue_opt, label="OPT")

ax[1].set_title("Rounds vs Reserve Price")
ax[1].plot(x, avg_reserve_price)

plt.show()
