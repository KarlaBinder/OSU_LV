
#radniSati=int(input('Radni sati: '))
#satnica=int(input('eura/h: '))

#plaća=radniSati*satnica
#print(f'Ukupno: {plaća}')

def total_euro(radniSati, satnica):
    return radniSati*satnica

radniSati=int(input('Radni sati: '))
satnica=int(input('eura/h: '))
print(f'Ukupno: {total_euro(radniSati,satnica)}')

