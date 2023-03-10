
#radniSati=float(input('Radni sati: '))
#satnica=float(input('eura/h: '))

#plaća=radniSati*satnica
#print(f'Ukupno: {plaća}')

def total_euro(radniSati, satnica):
    return radniSati*satnica

radniSati=float(input('Radni sati: '))
satnica=float(input('eura/h: '))
print(f'Ukupno: {total_euro(radniSati,satnica)}')

