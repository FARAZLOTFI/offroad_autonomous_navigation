from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
from models.nn_model import predictive_model_badgr
import numpy as np

class image_based_planner():
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

class rc_car_model:
    def __init__(self):
        self.C1= 0.5
        self.C2 = 10/6
        self.Cm1 = 12
        self.Cm2 = 2.5
        self.Cr2 = 0.15
        self.Cr0 = 0.7
        self.mu_m = 4.0
        self.g_ = 9.81
        self.dt = 0.2

    # where we use MHE to update the parameters each time we get a new measurements
    def state_parameters_update(self):
        pass
    def step(self, sigma, forward_throttle, dtheta):

        dX = V * np.cos(Sai + self.C1 * sigma)
        dY = V * np.sin(Sai + self.C1 * sigma)
        dSai = V * sigma * self.C2
        dV = (self.Cm1 - self.Cm2 * V * np.sign(V)) * forward_throttle - ((Cr2 * V ** 2 + self.Cr0) +
                                                                          (V * sigma)**2 * (self.C2 * self.C1 ** 2)) - \
                                                                            mu_m*g_*np.sin(Pitch)

        # The API does not support static augmentation, hence, I am using this simplified version
        dPitch = -0.000001*Pitch

        ddtheta = (tau - self.k * dtheta - self.m * self.g * self.l * math.sin(theta)) / (self.m * self.l * self.l)
        dtheta = dtheta + ddtheta * self.dt
        theta = theta + dtheta * self.dt
        return ddtheta, dtheta, theta

class MPC_controller:
    def __init__(self, system_model, num_of_states, num_of_actions, receding_horizon):
        self.system_model = system_model
        self.receding_horizon = receding_horizon
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.Q =  np.eye(self.num_of_states)# Weighting matrix for state trajectories
        self.R =  0.1*np.eye(self.num_of_actions)# Weighting matrix for control actions
        self.delta_R = np.eye(self.num_of_actions)
        self.stack_variables = np.ones(4)
        self.lambda_list = np.ones(4)
        self.last_action = np.zeros(self.num_of_actions)
    def MPC_cost(self, control_actions, theta_ref, max_actions, max_delta_actions, dtheta, theta):
        cost = 0
        states = np.array([0, dtheta, theta])

        for i in range(self.receding_horizon):
            states = self.system_model.step(control_actions[i], states[1], states[2])
            states_error = theta_ref - states[2]
            # constraint on the control action
            constraint_term1 = abs(control_actions[i] - max_actions)  # must be negative
            # contraint on the derivative of the control action
            if i>0:
                delta_actions = control_actions[i] - control_actions[i-1]
            else:
                delta_actions = control_actions[i] - self.last_action
            constraint_term2 = abs(delta_actions - max_delta_actions )    # positive

            cost += self.lambda_list[1]*constraint_term1 + self.lambda_list[2]*constraint_term2 + \
                    np.dot(np.dot(states_error,self.Q),states_error.transpose()) +\
                    np.dot(np.dot(control_actions[i],self.R),control_actions[i].transpose())
            #cost += states_error**2
        return cost
    def solve_mpc(self, control_actions, theta_ref, max_actions, max_delta_actions,dtheta, theta):
        optimal_actions = minimize(self.MPC_cost, control_actions,args=(theta_ref,max_actions,max_delta_actions,dtheta, theta))
        self.last_action = optimal_actions.x[0]
        return optimal_actions.x
class mpc_planner():
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

    steering_angle_planner = image_based_planner()

