r"""
Panel Method
============

This module provides the support for quickly and easily setting up and running the panel method.
It contains support for running a single case, or a sweep over different angles of attack.
"""
import numpy as np
import yaml
import time
import Modules.General as General
from Modules.Mesh import Mesh

# run single case
class SingleCase:
    r"""Manages and executes a single panel method simulation.

    Attributes:
        config (Config): Configuration object storing solver settings.
        mesh (Mesh): Mesh object containing the discretized geometry and solver routines.
    """
    def __init__(self, input_file):
        r"""Initializes the simulation environment.

        Args:
            input_file (str): Path to the main YAML configuration file.
        """
        # initialise setup
        self.config = General.Config(input_file)
        self.mesh = Mesh(self.config)

    def run(self):
        r"""Executes the panel method solver for the single configuration."""
        # run panel method
        self.mesh.run_case()

# run multiple cases
class Sweep:
    r"""Manages and executes a batch sweep of simulations over multiple angles of attack.

    Parses sweep parameters to generate an array of angles, overrides specific 
    configuration settings for batch execution (e.g., disabling visualizations), 
    and tracks solver progress and estimated time of completion.

    Attributes:
        sweep_data (dict): Raw dictionary of sweep parameters loaded from the YAML file.
        alphas (ndarray or list): Array of angle of attack values to simulate.
        total_cases (int): Total number of simulation iterations in the sweep.
        updates (bool): Flag to enable or disable progress printing in the console.
        config (Config): Configuration object initialized for batch processing.
        start_time (float): Timestamp marking the beginning of the sweep.
        last_time (float): Timestamp marking the completion of the most recent iteration.
        mesh (Mesh): The mesh and solver instance, rebuilt for each angle of attack.
    """
    def __init__(self, input_file, sweep_file):
        r"""Initializes the sweep parameters and sets up the baseline configuration.

        Args:
            input_file (str): Path to the main YAML configuration file.
            sweep_file (str): Path to the YAML file containing sweep bounds and settings.
        """
        # extract and define sweep data
        with open(sweep_file, "r") as file:
            self.sweep_data = yaml.safe_load(file)
        
        if self.sweep_data["alpha"]["type"] == 'explicit':
            self.alphas = self.sweep_data["alpha"]["values"]
        elif self.sweep_data["alpha"]["type"] == 'equal-spacing':
            self.alphas = np.arange(float(self.sweep_data["alpha"]["min_val"]), float(self.sweep_data["alpha"]["max_val"]) + float(self.sweep_data["alpha"]["spacing"]), float(self.sweep_data["alpha"]["spacing"]))
        elif self.sweep_data["alpha"]["type"] == 'number':
            self.alphas = np.linspace(float(self.sweep_data["alpha"]["min_val"]), float(self.sweep_data["alpha"]["max_val"]), float(self.sweep_data["alpha"]["number"]), endpoint=True)
        
        # get sweep information
        self.total_cases = len(self.alphas)
        self.updates = self.sweep_data["updates"]

        # load initial input file
        self.config = General.Config(input_file)

        # ensure results are saved and no visualisation shown
        self.config.sweep = True
        self.config.verbose = False
        self.config.write_to_file = True
        self.config.overwrite_output = False
        self.config.visualisation = False

    def run(self):
        r"""Executes the solver sequentially for all generated angles of attack.
        
        Iteratively updates the configuration, rebuilds the mesh, executes the solver, 
        and optionally prints timing and progress estimates to the console.
        """
        # take note of start time
        self.start_time = time.time()
        self.last_time = time.time()

        # print update
        if self.updates:
            print(f'\nStarting sweep through {self.total_cases} cases:\n')

        # loop through each case
        for i in range(self.total_cases):

            # update config for this case
            self.config.alpha = self.alphas[i]
            self.config.sweep_iteration = i
            self.config.update()
            
            # initialise new setup
            if i == 0:
                self.mesh = Mesh(self.config)
            else:
                self.mesh = Mesh(self.config, A=self.mesh.A)

            # run panel method
            self.mesh.run_case()

            # print update
            if self.updates:
                print(f'    Finished {i+1} of {self.total_cases} cases... (Time: {time.time()-self.last_time:.2}s, Time Left: {(time.time()-self.last_time) * (self.total_cases - (i+1)):.0f}s)')
                self.last_time = time.time()
        
        # print update
        if self.updates:
            print(f'\nDone ({self.last_time - self.start_time:.1f}s). Results saved to {self.config.out_coeff_file}\n')