from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
from models.nn_model import predictive_model_badgr
import numpy as np
from MHE_MPC.system_identification import MHE_MPC
class image_based_planner():
    def __init__(self,receding_horizon):
        self.CEM_initialization(receding_horizon)

    def CEM_initialization(self,receding_horizon):
        planning_horizon = receding_horizon
        self.planning_horizon = planning_horizon
        num_of_events = 3
        action_dimension = 2
        self.predictive_model = predictive_model_badgr(planning_horizon, num_of_events, action_dimension)
        self.events_rewards = torch.tensor(np.array([-1, -1, -1]), dtype=torch.float32)
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
        self.searcher = SNES(problem, stdev_init=1)

        # Create loggers as desired
        self.stdout_logger = StdOutLogger(self.searcher)  # Status printed to the stdout
        self.pandas_logger = PandasLogger(self.searcher)  # Status stored in a Pandas dataframe

    def objective(self, actions):
        # the output would be the probabilities and regression output
        self.actions[:,0,0] = actions
        events, bearings = self.predictive_model.predict_events(self.image_embedding, self.actions)

        loss = torch.tensor(0,dtype=torch.float32).cuda()
        for item in events:
            item = item[0,:]
            for i in range(len(item) - 1):
                loss += item[i]*self.events_rewards[i]

            # the last term stands for the direction difference with the goal

            loss += item[-1]*(1-torch.max(item[:3])) #TODO

        return loss

    def optimization_step(self, current_image=None):
        if current_image is not None: # if it's none, it means that another class is using this
            self.image_embedding = self.predictive_model.extract_features(current_image)

        uncertainties = torch.zeros(self.planning_horizon).cuda()
        # Run the algorithm for as many iterations as desired
        self.searcher.run(10) #TODO actions !!!
        #self.searcher.status['pop_best'].values.detach().cpu().numpy()
        # progress = self.pandas_logger.to_dataframe()
        # progress.mean_eval.plot()  # Display a graph of the evolutionary progress by using the pandas data frame
        return self.searcher.status['pop_best'].values, uncertainties
class rc_car_model:
    def __init__(self):
        # initial values of the parameters
        self.C1= torch.tensor(0.5)
        self.C2 = torch.tensor(10/6)
        self.Cm1 = torch.tensor(12)
        self.Cm2 = torch.tensor(2.5)
        self.Cr2 = torch.tensor(0.15)
        self.Cr0 = torch.tensor(0.7)
        self.mu_m = torch.tensor(4.0)
        self.g_ = torch.tensor(9.81)
        self.dt = torch.tensor(0.2)
        # the states initialization; we always assume that we've started from zero; then it's ok to add the
        # initial real world pose to the states to end up finding the realworld pose
        # self.X = 0
        # self.Y = 0
        # self.Sai = 0.0
        # self.V = 0
        # self.Pitch = 0
        self.states = torch.tensor(np.array([0, 0, 0, 0, 0]),dtype=torch.float32)
    # where we use MHE to update the parameters each time we get a new measurements
    def parameters_update(self, updated_parameters):
        self.C1,self.Cm1, self.Cm2, self.Cr2, self.Cr0, self.mu_m = torch.tensor(updated_parameters, dtype=torch.float32).cuda()

    def step(self, X, Y, Sai, V, Pitch, sigma, forward_throttle):
        X = V * torch.cos(Sai + self.C1 * sigma)
        Y = V * torch.sin(Sai + self.C1 * sigma)
        Sai = V * sigma * self.C2
        V = (self.Cm1 - self.Cm2 * V ) * forward_throttle - ((self.Cr2 * V ** 2 + self.Cr0) +
                                                                          (V * sigma)**2 * (self.C2 * self.C1 ** 2)) - \
                                                                            self.mu_m*self.g_*torch.sin(Pitch)
        Pitch = Pitch # static
        return X, Y, Sai, V, Pitch

class planner:
    def __init__(self, receding_horizon):
        self.system_model = rc_car_model()
        self.receding_horizon = receding_horizon
        self.num_of_states = 5
        self.num_of_actions = 1
        #self.Q =  np.eye(self.num_of_states)# Weighting matrix for state trajectories
        #self.R =  0.1*np.eye(self.num_of_actions)# Weighting matrix for control actions
        #self.delta_R = np.eye(self.num_of_actions)
        #self.stack_variables = np.ones(4)
        #self.lambda_list = np.ones(4)
        self.last_action = np.zeros(self.num_of_actions)
        # The estimator
        self.estimation_algorithm = MHE_MPC()
        # the optimization solver
        self.CEM_initialization()
        # image based planner
        self.steering_angle_planner = image_based_planner(receding_horizon)

    def CEM_initialization(self):
        action_dimension = 2
        self.actions = torch.rand(self.receding_horizon, 1, action_dimension)

        problem = Problem(
            "min",
            self.MPC_cost,
            initial_bounds=(-1, 1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.receding_horizon,
            # Higher-than-default precision
            dtype=torch.float32,
        )

        # Create a SearchAlgorithm instance to optimise the Problem instance
        self.searcher = SNES(problem, stdev_init=1)

        # Create loggers as desired
        self.stdout_logger = StdOutLogger(self.searcher)  # Status printed to the stdout
        self.pandas_logger = PandasLogger(self.searcher)  # Status stored in a Pandas dataframe

    def MPC_cost(self, set_of_throttles):

        # initialization of the mpc algorithm with the current states of the model
        states = self.system_model.states
        # I suppose in the optimization procedure we have an initial set of lin velocities/throttles
        self.steering_angle_planner.actions[:, 0, 1] = set_of_throttles

        # Now given the set of throttles the following step would be done
        set_of_steering_angles, uncertainties = self.steering_angle_planner.optimization_step()
        loss = torch.tensor([0],dtype=torch.float32).cuda()
        for i in range(self.receding_horizon):
            states = self.system_model.step(*states, set_of_steering_angles[i], set_of_throttles[i])
            # states_error = theta_ref - states[2]
            # constraint on the control action
            # constraint_term1 = abs(control_actions[i] - max_actions)  # must be negative
            # # contraint on the derivative of the control action
            # if i>0:
            #     delta_actions = control_actions[i] - control_actions[i-1]
            # else:
            #     delta_actions = control_actions[i] - self.last_action
            # constraint_term2 = abs(delta_actions - max_delta_actions )    # positive
            #################### use the traj planner to come up with a set of steering angles ########################


            # cost += self.lambda_list[1]*constraint_term1 + self.lambda_list[2]*constraint_term2 + \
            #         np.dot(np.dot(states_error,self.Q),states_error.transpose()) +\
            #         np.dot(np.dot(control_actions[i],self.R),control_actions[i].transpose())
            coef_vel = torch.tensor([1],dtype=torch.float32).cuda()
            coef_uncertainty = torch.tensor([10],dtype=torch.float32).cuda()
            # maximizing the velocity, minimizing the uncertainty

            loss += coef_vel/torch.tensor((states[3]**2).detach().cpu().numpy(),dtype=torch.float32).cuda() + coef_uncertainty*uncertainties[i]
        return loss
    def plan(self, current_image):
        observations = self.estimation_algorithm.measurement_update()
        # update the states
        self.system_model.states = torch.tensor(self.estimation_algorithm.mhe.make_step(observations),dtype=torch.float32).cuda()
        # update the parameters
        self.system_model.parameters_update(self.estimation_algorithm.mhe.data._p[-1])
        # Now process the image
        self.steering_angle_planner.image_embedding = self.steering_angle_planner.predictive_model.extract_features(current_image)

        self.searcher.run(10)
        return self.searcher.status['pop_best'].values

if __name__ == '__main__':
    current_image = torch.rand((1, 3, 128, 72))

    main_planner = planner(5)

    main_planner.plan(current_image)
    #steering_angle_planner = image_based_planner()

    #steering_angle_planner.optimization_step(current_image)