from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk
import seaborn as sns
import pandas as pd
from theano import tensor as T


df = pd.read_csv('C:/Users/hille/Desktop/Data science/Project/A-Game-of-Data---Data-Science-Exam-Project/survival.csv')
df.Dead = df.Dead.astype(np.int64)
#df.metastized = (df.metastized == 'yes').astype(np.int64)
n_patients = df.shape[0]
patients = np.arange(n_patients)


df = df.sort_values(by=['Nobility','time_of_monitoring'])

fig, ax = plt.subplots(figsize=(8, 6))

blue, _, red = sns.color_palette()[:3]

ax.hlines(patients[df.Dead.values == 0], 0, df[df.Dead.values == 0].time_of_monitoring,
          color=blue, label='Censored')

ax.hlines(patients[df.Dead.values == 1], 0, df[df.Dead.values == 1].time_of_monitoring,
          color=red, label='Uncensored/death')

ax.scatter(df[df.Nobility.values == 1].time_of_monitoring, patients[df.Nobility.values == 1],
           color='k', zorder=10, label='Noble')

ax.set_xlim(left=0)
ax.set_xlabel('Episodes since introduction')
ax.set_yticks([])
ax.set_ylabel('Character')

ax.set_ylim(-0.25, n_patients + 0.25)

ax.legend(loc='center right');



