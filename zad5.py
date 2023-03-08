file = open('SMSSpamCollection.txt', encoding="utf-8")
spam = 0
ham = 0
spamAverage = 0.0
hamAverage = 0.0
spamExclamation = 0

for line in file:
    line = line.rstrip()
    line = line.split()
    if line[0] == 'ham':
        ham += 1
        hamAverage += (len(line) - 1)
    else:
        spam += 1
        spamAverage += (len(line) - 1)
        if line[- 1].endswith('!'):
            spamExclamation += 1
file.close()

spamAverage /= spam
hamAverage /= ham

print(f'Average ham: {hamAverage}')
print(f'Average spam: {spamAverage}')
print(f'Total exclamation:{spamExclamation}')
