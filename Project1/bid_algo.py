import csv
import math
import numpy as np
 
def exactCalculation(bid, value, data):
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
  
 
def bidAlgo(data, orig, s = 50):
 
       s += len(data)
 
       opt = []
 
       for val in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
           bestBid = 0
           bestUtil = 0
           bestWinP = 0
 
           for i in range(10 * val):
               ans = exactCalculation(i * 0.1, val, data)
               if ans[1] > bestUtil:
                   bestUtil = ans[1]
                   bestBid = i * 0.1
                   bestWinP = ans[0]
 
           opt.append(bestBid)
 
      
       w1 = len(data) / s
  
       w2 = 1 - w1
 
       ans = []
 
       for i in range(len(opt)):
           ans.append(opt[i] * w1 + orig[i] * w2)
      
       return [ans, s]
 
 
with open('bid_data.csv', newline ='') as csv_file:
   csv_reader = csv.reader(csv_file, delimiter=',')
   my_data = list(csv_reader)
  
   my_bids = my_data[29]
   del my_data[29]
   del my_data[0]
 
   for i in range(len(my_bids)):
       my_bids[i] = float(my_bids[i])
 
   for i in range(len(my_data)):
       for j in range(len(my_data[i])):
           my_data[i][j] = float(my_data[i][j])
 
   # 10 possible values (for me) * 41 entries * 10 possible bids = 4100 possible bids
 
 
   win_p = []
   e_util = []
 
   for bid in my_bids:
       ans = exactCalculation(bid, bid + 9.9999, my_data)
       win_p.append(ans[0])
       e_util.append(ans[1])
 
 
   print("My win p: " + str(win_p))
   print("My expected utility: " + str(e_util) + "\n")
 
 
   opt_bids = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
 
   cur_s = 0
  
   ## giving all data at once
 
   [opt_bids, cur_s] = bidAlgo(my_data, opt_bids)
 
   print(opt_bids)
 
   opt_win_p = []
   opt_e_util = []
 
   for i in range(len(opt_bids)):
       ans = exactCalculation(opt_bids[i], (i + 1) * 10, my_data)
       opt_win_p.append(ans[0])
       opt_e_util.append(ans[1])
  
   print(opt_win_p)
   print(opt_e_util)
   print(sum(opt_e_util))
 
   opt_bids = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
 
   cur_s = 0
 
 
   ## online iterations (shuffle if needed)
 
   ## np.random.shuffle(my_data)
 
   for i in range(30):
       if cur_s == 0:
           [opt_bids, cur_s] = bidAlgo([my_data[i]], opt_bids)
       else:
           [opt_bids, cur_s] = bidAlgo([my_data[i]], opt_bids, cur_s)
      
 
   print(opt_bids)
 
   opt_win_p = []
   opt_e_util = []
 
   for i in range(len(opt_bids)):
       ans = exactCalculation(opt_bids[i], (i + 1) * 10, my_data)
       opt_win_p.append(ans[0])
       opt_e_util.append(ans[1])
  
   print(opt_win_p)
   print(opt_e_util)
   print(sum(opt_e_util))
