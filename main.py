import stanalyzer as sta


frame = sta.read_file('./tracks/S201409070000_E201409100000_VDBZc_T20_L5.pkl')

print(sta.life_cicle(frame))
