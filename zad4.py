counter=0
with open("song.txt","r") as file:
    dictionary={}
    for line in file:
        line=line.strip()
        line=line.lower()
        words=line.split()
        for word in words:
            if word in dictionary:
                dictionary[word] = dictionary[word] + 1
            else:
                dictionary[word]=1
    for key in list(dictionary.keys()):
        if dictionary[key]==1:
            counter+=1
            print(key,":",dictionary[key])
print(f'Total number:{counter}')



  
