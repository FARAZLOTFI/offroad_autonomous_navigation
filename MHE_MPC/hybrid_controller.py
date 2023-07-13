from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
from models.nn_model import predictive_model_badgr
import numpy as np

class planner():
    def __init__(self):
        self.CEM_initialization()

    def CEM_initialization(self):
        planning_horizon = 10
        num_of_events = 3
        action_dimension = 2
        self.predictive_model = predictive_model_badgr(planning_horizon, num_of_events, action_dimension)
        self.events_rewards = torch.tensor(np.array([-1, -1, -1]))
        self.image_embedding = None
        self.actions = torch.rand(planning_horizon, 1, action_dimension)

        problem = Problem(
            "min",
            self.objective,
            initial_bounds=(-1, 1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=planning_horizon,
            # Higher-than-default precision
            dtype=torch.float32,
        )

        # Create a SearchAlgorithm instance to optimise the Problem instance
        self.searcher = SNES(problem, stdev_init=5)

        # Create loggers as desired
        self.stdout_logger = StdOutLogger(self.searcher)  # Status printed to the stdout
        self.pandas_logger = PandasLogger(self.searcher)  # Status stored in a Pandas dataframe

    def objective(self, actions):
        # the output would be the probabilities and regression output
        self.actions[:,0,0] = actions
        events, beerings = self.predictive_model.predict_events(self.image_embedding, self.actions)

        loss = 0
        for item in events:
            item = item[0,:]
            for i in range(len(item) - 1):
                loss += item[i]*self.events_rewards[i]

            # the last term stands for the direction difference with the goal

            loss += item[-1]*(1-torch.max(item[:3])) #TODO

        return loss

    def optimization_step(self, current_image):
        self.image_embedding = self.predictive_model.extract_features(current_image)

        # Run the algorithm for as many iterations as desired
        self.searcher.run(10)

        progress = self.pandas_logger.to_dataframe()
        progress.mean_eval.plot()  # Display a graph of the evolutionary progress by using the pandas data frame

if __name__ == '__main__':
    current_image = torch.rand((1, 3, 128, 72))

    planner().optimization_step(current_image)
