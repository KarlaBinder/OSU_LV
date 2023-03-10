list = []

while True:
    try:
        n = input('Unesi broj: ')
        if n == 'Done':
            break
        list.append(float(n))
    except:
        print('Not a number')

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
print(f'Količina brojeva: {len(list)}, Srednja vrijednost: {average}, Minimalna vrijednost:{min}, Maksimalna vrijednost:{max}, Sortirana lista:{list}')







