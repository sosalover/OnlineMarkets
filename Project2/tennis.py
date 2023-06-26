# codes:
# Federer - 0, Nadal - 1, Djokovic - 2, Murray - 3, Wawrinka - 4 
# Gasquet - 5, Del Potro - 6, Ferrer - 7, Berdych - 8, Tsonga - 9
# Payoffs go in chron order, Aus 2008 (ind 0), French 2008, Wimb 2008, ..., US Open 2015 (ind 31)
# Fed 2008 payoff is payoff_data[0][0] + payoff_data[0][1] + payoff_data[0][2] + payoff_data[0][3]

import numpy as np

file = open("Project2/tennis_data.csv")

payoff_data = np.zeros([10, 32])

ind = 0

for line in file:
    data = line.split(",")
    del data[0]
    for data_ind, payoff in enumerate(data):
        payoff_data[ind % 10][int(ind / 10) + (4 * int(data_ind))] = int(payoff)
    ind += 1

    
for ind, data in enumerate(payoff_data):
    print(data)
