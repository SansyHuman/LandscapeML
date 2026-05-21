from common.inconsistents_parser import *


theories = ['SU2adj1nf1', 'SU2adj1nf2', 'Sp2adj1nf1', 'SO5adj1nf1', 'G2adj1nf1', 'SO5S1nf0',
            'SO5S1nf1', 'SO5S1nf2', 'Sp2adj2nf0', 'SO5adj1nf2', 'SO5nf5',
            'Sp2S1nf1', 'SU3adj1nf1', 'SU3adj1nf2', 'SU2nf4', 'SU2adj1nf3',
            'E_Usp6', 'SU5s2S2nf0', 'SU6s2S2nf0', 'SU6s2S2nf1', 'SU7s1S2a1f8nf0',
            'SU8s1S2a1f8nf1', 'SU5s1S1a1A1nf0', 'SU6s1S1a1A1nf0', 'SU5s1S1a1A1nf1',
            'SU5s1a1A2F8nf0', 'SU5s1a1A2F8nf1', 'SU9s2A2F16nf0', 'SU6s1a1A2F8nf1',
            'SU10s2A2F16nf1', 'SU5adj1s1S1nf0', 'SU6adj1s1S1nf0', 'SU5adj1s1S1nf1',
            'SU9a2A2nf0', 'SU5adj1s1A1F8nf0', 'SU7a2A2nf1', 'SU6adj1s1A1F8nf1',
            'SU9a2A2nf1', 'SU5adj1a1A1nf1', 'SU2adj2nf0', 'SU2adj2nf1', 'SU5s1S1a2A2nf0',
            'SU5s1S1a2A2nf1', 'SU5adj1a1A1nf0', 'SU5a3A3nf0', 'SU5a3A3nf1', 'Sp2adj1a2nf4',
            'SU5adj2a1A1nf0', 'SU5adj2a1A1nf1', 'SO11s2nf0', 'SO12s2nf1', 'SO7s1a1nf0',
            'SO8s1a1nf1', 'SO5a2nf0', 'SO5a2nf1', 'Sp2s2nf1', 'Sp4s1a1nf0', 'Sp2s1a1nf1',
            'Sp4s1a1nf1', 'Sp8a2nf0', 'Sp6a2nf1', 'Sp2s2a1nf1', 'Sp2s1a2nf0',
            'Sp2s1a2nf1', 'H0H1', 'SU3nf6', 'SU5adj1a2A2nf0', 'Sp4a3nf0', 'Sp3s2a1nf0',
            'Sp3a3nf1', 'SO5adj2nf1']

for theory in theories:
    vector = serialize_theory_name(theory)
    print(f'{theory}: {vector}')
    if vector[0] == 0:
        continue

    index_to_lie = {1: 'SU', 2: 'SO', 3: 'Sp', 4: 'SO', 5: 'E', 6: 'F', 7: 'G'}
    if not os.path.isdir(f'../inconsistents/{index_to_lie[vector[0]]}/{theory}'):
        print(f'{theory} directory does not exist')