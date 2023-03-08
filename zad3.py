def mean(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    average = sum/len(list)
    return average




list=[]
while True:
    n=input('Input a number: ')
    if n=='Done':
        print(f'Number count: {len(list)}, Minimal value: {min(list)}, Maximum value: {max(list)}')
    list.append(n)

list = []

while 1:
    try:
        temp = input('Unesi broj: ')
        if temp == 'Done':
            break
        list.append(float(temp))
    except:
        print('NaN exception')

print(len(list))
list.sort()
max = list[0]
min = list[0]
average = list[0]

for i in range(1, len(list)):
    if list[i] > max:
        max = list[i]
    if list[i] < min:
        min = list[i]
    average += list[i]

average /= len(list)
print(average)
print(max)
print(min)
print(list)






