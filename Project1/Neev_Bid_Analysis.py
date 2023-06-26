import csv
import matplotlib.pyplot as plt

with open('bid_data.csv', newline ='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv_reader)
    
    my_bids = data[29]
    del data[29]
    del data[0]

    for i in range(len(my_bids)):
        my_bids[i] = float(my_bids[i])

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])

    # 10 possible values (for me) * 41 entries * 10 possible bids = 4100 possible bids

    def exactCalculation(bid, value):
        wins = 0
        utility = 0

        for i in range(len(data)):
            for j in range(len(data[i])):
                if bid > data[i][j]:
                    wins += 1
                    utility += (value - bid)
                elif bid == data[i][j]:
                    wins += 0.5
                    utility += (value - bid) / 2
        return [wins / 410, utility / 410]

    win_p = []
    e_util = []

    for bid in my_bids:
        ans = exactCalculation(bid, bid + 9.9999)
        win_p.append(ans[0])
        e_util.append(ans[1])

    print("My win p: " + str(win_p))
    print("My expected utility: " + str(e_util))
    
    

    opt_bids = []
    opt_e_util = []
    opt_win_p = []

    for val in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        bestBid = 0
        bestUtil = 0
        bestWinP = 0
        
        utilities = []
        r = [x * 0.01 for x in range(1, val*100)]
        for j in r:
            ans = exactCalculation(j, val)
            utilities.append(ans[1])
            if ans[1] > bestUtil:
                bestUtil = ans[1]
                bestBid = j
                bestWinP = ans[0]

        opt_bids.append(bestBid)
        opt_e_util.append(bestUtil)
        opt_win_p.append(bestWinP)
        plt.plot(r, utilities)
        plt.xlabel ('Bid')
        plt.ylabel ('Utility')
        plt.title ('Utility for Bids')
    plt.savefig('exact_bid_total_{}')
    
    print("My bids: " + str(my_bids))
    print("Optimal bids: " + str(opt_bids) + "\n")
    print(str(opt_win_p))
    print(str(opt_e_util) + "\n")

    print(sum(e_util) / 10)
    print(sum(opt_e_util)/ 10)
