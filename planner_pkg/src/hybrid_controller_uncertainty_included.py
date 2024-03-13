#!/offroad_nav/hybrid_planner/bin/python
import cv2
from evotorch import Problem
from evotorch.algorithms import SNES, CEM, CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
from src.offroad_autonomous_navigation.models.nn_model import PredictiveModelBadgr, LSTMSeqModel, TransformerSeqModel
import os
import numpy as np
from src.offroad_autonomous_navigation.MHE_MPC.system_identification import MHE_MPC
from src.offroad_autonomous_navigation.traj_planner_helpers import load_data_lists, input_preparation_practice, total_loss
from src.offroad_autonomous_navigation.metrics import Metrics
from tqdm import tqdm
from torchvision import transforms
# ROS dealing with the images
import rospy
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
import config
br = CvBridge()

image_size=(72,128)

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.ToTensor()
    ])
def callback_image(imgdata):
    global rawImage
    rawImage = br.imgmsg_to_cv2(imgdata,"bgr8")
    # cv2.imshow('realtime image',rawImage)
    # cv2.waitKey(1)
def callback_gps(data):
    global gps_data
    global flag_gps_data
    gps_data = [data.longitude, data.latitude]
    flag_gps_data = True

def callback_imu(data):
    global robot_orientation
    robot_orientation = data.orientation

class image_based_planner():
    def __init__(self,receding_horizon, optimization_horizon):
        self.planning_horizon = receding_horizon
        self.vision_optimization_horizon = optimization_horizon
        self.CEM_initialization()

        self.temporal_rewards = 0.0
        self.metrics = Metrics(planning_horizon=receding_horizon, device=self.device)
    def model_initialization(self):
        num_event_types = 9 + 1  # one for regression
        n_seq_model_layers = 4
        seq_elem_dim = 16
        action_dimension = 2
        # seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
        seq_encoder = TransformerSeqModel(n_seq_model_layers, seq_elem_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictive_model  = PredictiveModelBadgr(self.planning_horizon, num_event_types,
                                     action_dimension, seq_encoder, n_seq_model_layers, device = self.device, ensemble_size=5)
        self.predictive_model .cuda()

        CHECKPOINT_PATH = config.weights_file

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.predictive_model .load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.predictive_model.eval()

        self.model_uncertainties = torch.zeros(self.planning_horizon).cuda()
        self.desired_heading = torch.tensor(np.zeros([self.planning_horizon,1]), dtype=torch.float32).cuda()
    def CEM_initialization(self):

        num_of_events = 3
        action_dimension = 2

        self.model_initialization()

        self.events_rewards = torch.tensor(np.array([[-1], [-1], [-1], [-1], [-1], [-1], [0.8],[1],[0.8]]), dtype=torch.float32).cuda()
        self.image_embedding = None
        self.current_image = None
        self.actions = torch.rand(self.planning_horizon, 1, action_dimension, device="cuda:0")

        problem = Problem(
            "max",
            self.objective,
            initial_bounds=(0, 0),
            bounds=(-1 ,1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.vision_optimization_horizon,
            # Higher-than-default precision
            dtype=torch.float32,
        )

        # Create a SearchAlgorithm instance to optimise the Problem instance
        self.searcher = CMAES(problem, popsize=2, stdev_init=1)

        # Create loggers as desired
        self.stdout_logger = StdOutLogger(self.searcher)  # Status printed to the stdout
        self.pandas_logger = PandasLogger(self.searcher)  # Status stored in a Pandas dataframe

    def objective(self, actions):
        # the input must be filtered
        steering_actions = torch.zeros(self.planning_horizon).cuda()
        for i in range(self.vision_optimization_horizon):
            steering_actions[i] = actions[i]

        #actions = torch.cat((actions,torch.zeros(self.planning_horizon - self.optimization_horizon).cuda()))
        self.actions[:,0,0] = torch.tanh(steering_actions)*(-0.6)
        ##########################################
        inputs_ = [self.current_image, self.actions]
        classification_output, pred_regression, epi_unc_classification, epi_unc_regressions = self.metrics.calc_unc(self.predictive_model, inputs=inputs_) # TODO BUG
        pred_classification, mix_dist = classification_output
        self.model_uncertainties = epi_unc_classification
        ##events, bearings = self.predictive_model.predict_events(self.image_embedding, self.actions)
        events = pred_classification
        loss = torch.tensor([0],dtype=torch.float32).cuda()

        # puting the variables on the GPU
        pred_classification = torch.tensor(pred_classification, dtype=torch.float32).cuda()
        pred_regression = torch.tensor(pred_regression, dtype=torch.float32).cuda()
        mix_dist = torch.tensor(mix_dist, dtype=torch.float32).cuda()

        for i,event in enumerate(events):
            # each event stands for the probability of that event to happen
            loss += self.events_rewards[event]
            loss += (pred_regression[i] - self.desired_heading[i]) * (torch.ones(1).cuda() - torch.max(mix_dist[i][0][:3]))  # in either case of {tree,other_obstacles,human}, we don't care about the heading


        self.temporal_rewards = loss

        return loss
    def optimization_step(self, current_image=None):
        ##if current_image is not None: # if it's none, it means that another class is using this TODO
        ##    self.image_embedding = self.predictive_model.extract_features(current_image) TODO

        ##uncertainties = torch.zeros(self.planning_horizon).cuda()

        # Run the algorithm for as many iterations as desired
        self.searcher.run(1) #TODO actions !!!
        #self.searcher.status['pop_best'].values.detach().cpu().numpy()
        # progress = self.pandas_logger.to_dataframe()
        # progress.mean_eval.plot()  # Display a graph of the evolutionary progress by using the pandas data frame
        return self.searcher.status['pop_best'].values, self.model_uncertainties
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
        self.states = torch.tensor(np.array([0, 0, 0, 0, 0]),dtype=torch.float32).cuda()
    # where we use MHE to update the parameters each time we get a new measurements
    def parameters_update(self, updated_parameters):
        self.C1,self.Cm1, self.Cm2, self.Cr2, self.Cr0, self.mu_m = torch.tensor(updated_parameters, dtype=torch.float32).cuda()

    def step(self, X, Y, Sai, V, Pitch, sigma, forward_throttle):
        sigma = torch.tanh(sigma)*(-0.6)
        forward_throttle = (torch.tanh(forward_throttle) +0.5)/1.5 # -0.33 ~ 1 not to maximize the speed on the reverse side
        X = (V * torch.cos(Sai + self.C1 * sigma))*self.dt + X
        Y = (V * torch.sin(Sai + self.C1 * sigma))*self.dt + Y
        Sai = (V * sigma * self.C2)*self.dt + Sai
        V = ((self.Cm1 - self.Cm2 * V ) * forward_throttle - ((self.Cr2 * V ** 2 + self.Cr0) + \
                    (V * sigma)**2 * (self.C2 * self.C1 ** 2)) - self.mu_m*self.g_*torch.sin(Pitch))*self.dt + V
        Pitch = Pitch # static

        return X, Y, Sai, V, Pitch

class planner:
    def __init__(self, vision_model_receding_horizon, moh, mpcoh, gps_initial_data):
        self.system_model = rc_car_model()
        self.receding_horizon = vision_model_receding_horizon
        self.mpc_optimization_horizon = mpcoh
        self.vision_optimization_horizon = moh
        self.num_of_states = 5
        self.num_of_actions = 1
        #self.Q =  np.eye(self.num_of_states)# Weighting matrix for state trajectories
        #self.R =  0.1*np.eye(self.num_of_actions)# Weighting matrix for control actions
        #self.delta_R = np.eye(self.num_of_actions)
        #self.stack_variables = np.ones(4)
        #self.lambda_list = np.ones(4)
        self.last_action = np.zeros(self.num_of_actions)
        # The estimator
        self.estimation_algorithm = MHE_MPC(GPS_initial_data=gps_initial_data)
        # the optimization solver
        self.CEM_initialization()
        # image based planner
        self.steering_angle_planner = image_based_planner(self.receding_horizon, self.vision_optimization_horizon)
        self.planned_steering_angles = 0
        self.speed = []
    def CEM_initialization(self):
        action_dimension = 2
        self.actions = torch.rand(self.receding_horizon, 1, action_dimension)

        problem = Problem(
            "min",
            self.MPC_cost,
            initial_bounds=(0, 0),
            bounds=(-1, 1),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            solution_length=self.mpc_optimization_horizon,
            # Higher-than-default precision
            dtype=torch.float32,
        )

        # Create a SearchAlgorithm instance to optimise the Problem instance
        self.searcher = CMAES(problem, popsize=2, stdev_init=1)

        # Create loggers as desired
        self.stdout_logger = StdOutLogger(self.searcher)  # Status printed to the stdout
        self.pandas_logger = PandasLogger(self.searcher)  # Status stored in a Pandas dataframe

    def MPC_cost(self, set_of_throttles):

        # initialization of the mpc algorithm with the current states of the model
        states = self.system_model.states
        # I suppose in the optimization procedure we have an initial set of lin velocities/throttles
        # instead of 40 actions just take the first 5 actions
        throttle_actions = torch.zeros(self.receding_horizon).cuda()
        for i in range(self.mpc_optimization_horizon):
            throttle_actions[i] = set_of_throttles[i]

        self.steering_angle_planner.actions[:, 0, 1] = throttle_actions

        # Now given the set of throttles the following step would be done
        set_of_steering_angles, uncertainties = self.steering_angle_planner.optimization_step()
        uncertainties = uncertainties.cuda()
        loss = torch.zeros(self.receding_horizon).cuda()#torch.tensor([0],dtype=torch.float32).cuda()
        self.speed = []
        for i in range(self.mpc_optimization_horizon): # TODO make sure that this horizon is shorter than the vision one
            states = self.system_model.step(*states, set_of_steering_angles[i], throttle_actions[i])

            coef_vel = torch.tensor([1],dtype=torch.float32).cuda()
            coef_uncertainty = torch.tensor([10],dtype=torch.float32).cuda()
            # maximizing the velocity, minimizing the uncertainty
            # coef_vel/torch.tensor((states[3]**2).detach().cpu().numpy(),dtype=torch.float32).cuda() + coef_uncertainty*uncertainties[i]
            loss[i] = coef_vel/(states[3]**2 + 1) + coef_uncertainty*uncertainties[i]
            # to record the speeds
            self.speed.append(states[3])
        # to record the speeds for further evaluation
        self.speed = torch.stack(self.speed).sum() / self.receding_horizon
        self.planned_steering_angles = set_of_steering_angles
        return loss.sum()
    def plan(self, current_image, sensors_data):
        observations = self.estimation_algorithm.measurement_update(sensors_data)
        # update the states
        self.system_model.states = torch.tensor(self.estimation_algorithm.mhe.make_step(observations),dtype=torch.float32).cuda()
        # update the parameters
        self.system_model.parameters_update(self.estimation_algorithm.mhe.data._p[-1])
        # Now process the image
        ###self.steering_angle_planner.image_embedding = self.steering_angle_planner.predictive_model.extract_features(current_image) TODO
        self.steering_angle_planner.current_image = current_image

        self.searcher.run(1)

        #self.debug_(self.steering_angle_planner.searcher.status['pop_best'].values, self.searcher.status['pop_best'].values)
        return self.searcher.status['pop_best'].values, self.planned_steering_angles

    def debug_(self, set_of_steering_angles, set_of_throttles): #TODO
        states = self.system_model.states
        for i in range(self.receding_horizon):
            states = self.system_model.step(*states, set_of_steering_angles[i], set_of_throttles[i])
            print('Action: ',[set_of_steering_angles[i].detach().cpu(),set_of_throttles[i].detach().cpu()],' *states: ', states)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flag_gps_data = False

    list_actions = [ ]
    uncertainties = [ ]
    recorded_rewards = []
    recorded_speeds = []
    count_sample = 0

    ### ROS initialization ####
    rospy.init_node('hybrid_planner', anonymous=True)

    imageTopicPath = config.imageTopic
    GPSTopicPath = config.GPSTopic
    IMUTopicPath = config.IMUTopic

    rospy.Subscriber(imageTopicPath, Image, callback_image)
    rospy.Subscriber(GPSTopicPath, NavSatFix, callback_gps)
    rospy.Subscriber(IMUTopicPath, Imu, callback_imu)

    rate = rospy.Rate(50)
    ############################
    rawImage = None

    while (not flag_gps_data):
        time.sleep(0.5)
        print('Waiting for the GPS data...')
    print('GPS locked!')
    print('GPS location: ', gps_data)
    BATCH_SIZE = 1
    planning_horizon = 40
    mpc_planner_horizon = 40
    vision_planner_horizon = 40
    main_planner = planner(planning_horizon, moh=vision_planner_horizon, mpcoh=mpc_planner_horizon, gps_initial_data=gps_data)
    # initialize the commands
    steering_angle = 0
    throttle = 0
    while not rospy.is_shutdown():
        if rawImage is not None:
            begin_time = time.time()
            current_image = input_preparation_practice(rawImage, augment=augment, debug_=False)
            if len(current_image)==0:
                continue

            given_measurements = [robot_orientation, gps_data, steering_angle, throttle]
            proposed_throttles, proposed_steering_angles = main_planner.plan(current_image, given_measurements)
            list_actions.append(proposed_throttles)
            throttle = proposed_throttles[0].detach().cpu().numpy()
            steering_angle = proposed_steering_angles[0].detach().cpu().numpy()
            uncertainties.append(main_planner.steering_angle_planner.model_uncertainties)
            recorded_rewards.append(main_planner.steering_angle_planner.temporal_rewards)
            recorded_speeds.append(main_planner.speed)

            count_sample +=1
        rate.sleep()
