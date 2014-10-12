import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NFLIPS = 10
SMP_SIZE = 1000
EXP_SIZE = 100000

v_1 = []
v_rand = []
v_min = []
for i in range(EXP_SIZE):
	df = pd.DataFrame(np.random.random_integers(0,1,(SMP_SIZE,NFLIPS)))
	df['v'] = df.apply(sum,axis=1)/10.
	v_1_i = df['v'][0]
	rand_i = np.random.random_integers(0,SMP_SIZE-1)
	v_rand_i = df['v'][rand_i]
	v_min_i = min(df['v'])
	
	v_1.append(v_1_i)
	v_rand.append(v_rand_i)
	v_min.append(v_min_i)

print np.mean(v_1), np.mean(v_rand), np.mean(v_min)

f,ax = plt.subplots(3)
ax[0].hist(v_1)
ax[1].hist(v_rand)
ax[2].hist(v_min)

plt.show()

