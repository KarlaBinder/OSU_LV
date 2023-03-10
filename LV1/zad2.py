
try:  
    n=float(input("Type a number: "))
    if n<0.0 or n>1.0:
        print('Out of range.')
    elif n<0.6:
        print('F')
    elif n>=0.9:
        print('A')
    elif n>=0.8:
        print('B')
    elif n>=0.7:
        print('C')
    elif n>=0.6:
        print('D')
except:
    print("Not a number.") 


