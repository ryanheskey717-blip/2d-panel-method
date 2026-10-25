r"""
Sweep Analysis
==============

This module provides the support for analysing the results of a sweep. It can load a results file and
plot any of the contained parameters. This module can also load XFOIL results to be used for comparison
to generated data.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

class Results:
    r"""Parses and plots aerodynamic simulation results from a CSV file.

    Attributes:
        data (dict): Dictionary mapping CSV column headers to lists of parsed data. 
            Numeric columns are stored as floats, while text columns remain as strings.
    """
    def __init__(self, filename):
        r"""Initializes the Results object by reading and parsing a CSV file.

        Args:
            filename (str): Path to the CSV file containing the results.
        """
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
        r"""Generates a single line plot of specified variables.

        Args:
            x (str): The dictionary key for the independent variable (x-axis).
            y (str): The dictionary key for the dependent variable (y-axis).
            show (bool, optional): Flag to display the plot immediately. Defaults to `True`.
            **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.plot`.
        """
        plt.figure()
        plt.plot(self.data[x], self.data[y], marker='o', label='2D Panel Method', **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(ls=':')
        plt.legend()
        if show:
            plt.show()
        
    def subplots(self, x, y, nrows, ncols, compare=None, show=True, **kwargs):
        r"""Generates a grid of subplots for multiple variables.

        Args:
            x (list[str]): List of dictionary keys for the x-axis of each subplot.
            y (list[str]): List of dictionary keys for the y-axis of each subplot.
            nrows (int): Number of rows in the subplot grid.
            ncols (int): Number of columns in the subplot grid.
            compare (object, optional): Another results object (e.g., `XFOILResults`) to 
                plot alongside the primary data for comparison. Defaults to `None`.
            show (bool, optional): Flag to display the plot immediately. Defaults to `True`.
            **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.plot`.
        """
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
    r"""Parses an XFOIL polar output text file.

    Attributes:
        data (dict): Dictionary mapping standard XFOIL headers to lists of float values.
    """
    def __init__(self, filename):
        r"""Initializes the XFOILResults object by reading a polar text file.

        Args:
            filename (str): Path to the XFOIL polar text file.
        """
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