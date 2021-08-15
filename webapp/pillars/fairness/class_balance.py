def compute_class_balance(data):
    print("hi")
    print(data)
    

'''
from scipy.stats import chisquare

significance_level = 0.05
p_value = chisquare(absolute_class_occurences, ddof=0, axis=0).pvalue

print("P-Value of Chi Squared Test: {0}".format(p_value))
if p_value < significance_level:
    print("The data does not follow a unit distribution")
else:
    print("We can not reject the null hypothesis assuming that the data follows a unit distribution")
    
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11);
xU, xL = x + 1, x - 1;
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3);
prob = prob / prob.sum(); # normalize the probabilities so their sum is 1
sample = np.random.choice(x, size = 10000, p = prob);
(_, observed_occurences) = np.unique(sample, return_counts=True)
plt.hist(nums, bins = len(x))

p_value = chisquare(observed_occurences, ddof=0, axis=0).pvalue

print("P-Value of Chi Squared Test: {0}".format(p_value))

if p_value < significance_level:
    print("The data does not follow a unit distribution")
else:
    print("We can not reject the null hypothesis assuming that the data follows a unit distribution")
'''