import numpy as np

revenue = np.array([[100,200,220],[24,36,40],[12,18,20]])
expenses = np.array([[80,90,100],[10,16,20],[8,10,10]])

profit = revenue - expenses
print(profit)

price_per_unit = np.array([1000,400,1200])
units = np.array([[30,40,50],[5,10,15],[2,5,7]])

# Broadcasting
rev1 = price_per_unit * units
print(rev1)

# Dot product
rev2 = price_per_unit.dot(units)
print(rev2)

# Exercise --> Q1
comp = np.array([[200,220,250],[68,79,105],[110,140,180],[80,85,90]])
comp_rupee = comp*75
print(comp_rupee)

# Exercise --> Q2
price_per_unit = np.array([20,30,15])
units = np.array([[50,60,25],[10,13,5],[40,70,52]])

rev = price_per_unit.dot(units)
print(rev)