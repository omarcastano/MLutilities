import numpy as np

def cramerv_relationship_strength(degrees_of_freedom, cramerv):
   """
   returns the strength of the relationship of two categorical variables 

   source: https://www.statology.org/interpret-cramers-v/
   """
   values = {
       "1": [0.10, 0.50],
       "2": [0.07, 0.35],
       "3": [0.06, 0.29],
       "4": [0.05, 0.25],
       "5": [0.04, 0.22]
   }

   if np.round(cramerv, 2) <= values[str(degrees_of_freedom)][0]:
     return "small"
   elif np.round(cramerv, 2) >= values[str(degrees_of_freedom)][-1]:
     return "high"
   else:
     return "medium"
