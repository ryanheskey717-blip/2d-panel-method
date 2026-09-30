import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

class Results:
    def __init__(self, filename):

        # initialise empty dictionary
        data = defaultdict(list)

        # open and read file
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                for header, value in row.items():
                    try: # all numeric values
                        data[header].append(float(value))
                    except ValueError: # any actual words (first column)
                        data[header].append(value)

        # convert to regular dictionary
        self.data = dict(data)

    def plot(self, x, y):
        plt.figure()
        plt.plot(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()