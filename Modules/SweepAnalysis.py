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

    def plot(self, x, y, show=True, **kwargs):
        plt.figure()
        plt.plot(self.data[x], self.data[y], marker='o', label='2D Panel Method', **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(ls=':')
        plt.legend()
        if show:
            plt.show()
        
    def subplots(self, x, y, nrows, ncols, compare=None, show=True, **kwargs):
        plt.figure(figsize=(12, 5))
        for i in range(nrows * ncols):
            plt.subplot(nrows, ncols, i+1)
            plt.plot(self.data[x[i]], self.data[y[i]], marker='o', label='2D Panel Method', **kwargs)
            if compare is not None:
                plt.plot(compare.data[x[i]], compare.data[y[i]], label='XFOIL Prediction', **kwargs)
            plt.xlabel(x[i])
            plt.ylabel(y[i])
            plt.grid(ls=':')
            plt.legend()
        plt.tight_layout(pad=2.0)
        if show:
            plt.show()

class XFOILResults:
    def __init__(self, filename):
        """Parses an XFOIL polar text file and returns a dictionary of column data."""
        data = {}
        headers = ['alpha_deg', 'c_l', 'c_d', 'CDp', 'c_m', 'Top_Xtr', 'Bot_Xtr']
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        parsing_data = False
        
        for line in lines:
            # detect table headers (identified by the line preceding the dashed line '------')
            if '------' in line:
                parsing_data = True
                continue
                
            if not parsing_data:
                # look for header line right before the dashed separator
                if 'alpha' in line and 'CL' in line:
                    #headers = line.strip().split()
                    data = {header: [] for header in headers}
            if parsing_data:
                # parse numerical rows
                parts = line.strip().split()
                if len(parts) == len(headers):
                    for header, val in zip(headers, parts):
                        data[header].append(float(val))
                        
        self.data = data