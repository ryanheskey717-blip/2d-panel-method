# import modules
import numpy as np
import yaml
import Modules.General as General
from Modules.Mesh import Mesh

# run single case
class SingleCase:

    def __init__(self, input_file):

        # initialise setup
        self.config = General.Config(input_file)
        self.mesh = Mesh(self.config)

    def run(self):

        # run panel method
        self.mesh.run_case()

# run multiple cases
class Sweep:

    def __init__(self, input_file, sweep_file):

        # extract sweep data
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

        # load initial input file
        self.config = General.Config(input_file)

        # ensure results are saved and no visualisation shown
        ###### TODO: self.config.write_to_file = True
        self.config.visualisation = False

    def run(self):

        # print update
        if self.sweep_data["updates"]:
            print(f'\nStarting sweep through {self.total_cases} cases:\n')

        # loop through each case
        for i in range(self.total_cases):

            # update config for this case
            self.config.alpha = self.alphas[i]
            self.config.update()
            
            # initialise new setup
            self.mesh = Mesh(self.config)

            # run panel method
            self.mesh.run_case()

            # print update
            if self.sweep_data["updates"]:
                print(f'    Finished {i+1} of {self.total_cases} cases...')
        
        # print update
        if self.sweep_data["updates"]:
            print(f'\nDone. Results saved to {self.config.out_coeff_file}\n')