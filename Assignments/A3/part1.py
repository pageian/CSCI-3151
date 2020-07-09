'''
    Gaussian Mixture Model for red blood cell counts in sample
    
    Males:
        Mean - 5.3m cells/uL
        STD  - 0.35m cells/uL

    Females:
        Mean - 4.6m cells/uL
        STD  - 0.2m cells/uL

    Only 40% of donations made by men
    Generate pop of 10,000
'''

import numpy as np
import matplotlib.pyplot as plt

# Variable Definitions
male_mu    = 5.3
male_std   = 0.35
female_mu  = 4.6
female_std = 0.2

male_ratio = 0.4
pop_count = 10000

# Generate Data
male_pop = np.random.normal(male_mu, male_std, int(male_ratio * pop_count))
female_pop = np.random.normal(female_mu, female_std, int(pop_count - (male_ratio * pop_count)))

# Plot Data
plt.hist(male_pop, density=False, color='b', alpha = 0.35, label='Male')
plt.hist(female_pop, density=False, color='r', alpha = 0.35, label='Female')
plt.legend()
plt.xlabel('Million Cells / uL')
plt.ylabel('Counts of Indiviuals')

plt.show()