def adversarial_data_gen(rounds, actions):
    data = []
    for round in range(rounds):
        data.append([])
        for action in range(actions):
            data[len(data) - 1].append(int((round % actions) == action) * (round + 1))
    return data

ans = adversarial_data_gen(10, 4)

for a in ans:
    print(a)
