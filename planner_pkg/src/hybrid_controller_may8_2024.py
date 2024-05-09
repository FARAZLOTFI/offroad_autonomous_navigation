#!/mnt/5ff2307d-f850-4368-8eeb-180d89716a8a/offroad_nav/hybrid_planner/bin/python
import os
import sys
##import matplotlib.pyplot as plt
##plt.ion()
# Redirect stdout to /dev/null
#sys.stdout = open(os.devnull, 'w')
import cv2
from src.offroad_autonomous_navigation.planner_pkg.src.customized_optimizerv2 import CEM_optimizer
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
from sensor_msgs.msg import Image, NavSatFix, Imu
from rccar_controller_pc.msg import TeensySerial, RCCarCommand
from cv_bridge import CvBridge
import config
br = CvBridge()
# this we use instead of the standard subscribers to sync our multi-modal data
import message_filters

image_size=(72,128)

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.ToTensor()
    ])

def debugDisplayer(img):
    global planner_command
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 440)
    fontScale = 0.7
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(planner_command),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    # Display the image
    ###cv2.namedWindow('realtime image', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    ###cv2.resizeWindow('realtime image', 400, 300)
    cv2.imshow('realtime image', img)
    key_ = cv2.waitKey(1)
    if (key_ == ord('q')):
        cv2.destroyAllWindows()

def decPlace(value):
    return round(value*1000)/1000

def callbackImage(imgdata):
    global raw_image
    raw_image = br.imgmsg_to_cv2(imgdata,"bgr8")
    #debugDisplayer(raw_image)

# This we use to get command from the RC controller
def callbackTeensy(teensy):
    global teensy_data
    teensy_data = teensy
def callbackImu(imu):
    global robot_orientation
    robot_orientation = imu.orientation

def callbackGPS(gps):
    global gps_data
    global flag_gps_data
    gps_data = [gps.longitude, gps.latitude]
    flag_gps_data = True
# def callbackSynced(imgdata, teensy, imu):
#     # for some reason syncing GPS with these data did not work
#     # for imu
#     global robot_orientation
#     robot_orientation = imu.orientation
#     # for teensy
#     global teensy_data
#     teensy_data = teensy
#     # for Image
#     global raw_image
#     raw_image = br.imgmsg_to_cv2(imgdata, "bgr8")
#     debugDisplayer(raw_image)
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
    def __init__(self, vision_model_receding_horizon, moh, mpcoh, gps_initial_data, vision_initial_pop_size=150,
                 vision_second_pop_size=45,
                 steering_angle_actions_list=[-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8],
                 throttle_actions_list=[0.2, 0.5, 0.8]):
        self.system_model = rc_car_model()
        self.receding_horizon = vision_model_receding_horizon
        self.mpc_optimization_horizon = mpcoh
        self.vision_optimization_horizon = moh
        self.vision_initial_pop_size = vision_initial_pop_size
        self.vision_second_pop_size = vision_second_pop_size
        self.num_of_states = 5
        self.num_of_actions = 1
        self.last_action = np.zeros(self.num_of_actions)
        # The estimator
        self.estimation_algorithm = MHE_MPC(GPS_initial_data=gps_initial_data)
        # image based planner
        self.speed = []

        self.steering_angle_actions_list = steering_angle_actions_list
        self.throttle_actions_list = throttle_actions_list
        if not self.vision_initial_pop_size%len(self.throttle_actions_list) ==0:
            print('Please change the BATCHSIZE value to be dividable by: ',len(throttle_actions_list))
            input('')

        self.vision_module_initialization()

    def vision_module_initialization(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flag_gps_data = False

        num_event_types = 9 + 1  # one for regression
        n_seq_model_layers = 4
        seq_elem_dim = 16
        action_dimension = 2
        # seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
        seq_encoder = TransformerSeqModel(n_seq_model_layers, seq_elem_dim)
        predictive_model = PredictiveModelBadgr(self.receding_horizon, num_event_types,
                                                     action_dimension, seq_encoder, n_seq_model_layers, device=device,
                                                     ensemble_size=3)
        predictive_model.cuda()
        path_base_directory = '/mnt/5ff2307d-f850-4368-8eeb-180d89716a8a/offroad_nav/planner_ws/src/offroad_autonomous_navigation'
        CHECKPOINT_PATH = path_base_directory + '/weights/training_checkpoint_epoch584.pt' # 250
        #
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        predictive_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        predictive_model.eval()
        self.steering_angle_planner = CEM_optimizer(self.steering_angle_actions_list,
                                                    self.throttle_actions_list,
                                                    predictive_model,
                                                    batch_size=self.vision_initial_pop_size,
                                                    initial_pop_size= self.vision_initial_pop_size,
                                                    second_pop_size=self.vision_second_pop_size,
                                                    planning_horizon=self.vision_optimization_horizon,
                                                    model_receding_horizon=self.receding_horizon)

    def MPC_search(self, set_of_throttles):

        # initialization of the mpc algorithm with the current states of the model
        states = self.system_model.states
        # I suppose in the optimization procedure we have an initial set of lin velocities/throttles
        # instead of 40 actions just take the first 5 actions
        # steering angles are given already!
        set_of_steering_angles = self.proposed_steering_angles

        best_throttle_index = 0
        previous_loss = None
        for count, (set_of_throttles, uncertainties) in enumerate(zip(self.selected_throttles, self.uncertainty_list)): #TODO from here
            throttle_actions = torch.zeros(self.receding_horizon).cuda()
            throttle_actions[:self.mpc_optimization_horizon] = set_of_throttles[:self.mpc_optimization_horizon]


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
            # for debugging
            ##print('loss '+str(count) + ': ',loss.sum())
            # to record the speeds for further evaluation
            self.speed = torch.stack(self.speed).sum() / self.receding_horizon
            if previous_loss == None:
                previous_loss = loss.sum()
            else:
                if previous_loss>loss.sum():
                    previous_loss = loss.sum()
                    best_throttle_index = count
        return self.selected_throttles[best_throttle_index]
    def plan(self, current_image, sensors_data):
        ####################################################### sys identification processing time is 0.014
        observations = self.estimation_algorithm.measurement_update(sensors_data)
        # update the states
        self.system_model.states = torch.tensor(self.estimation_algorithm.mhe.make_step(observations),dtype=torch.float32).cuda()
        # update the parameters
        self.system_model.parameters_update(self.estimation_algorithm.mhe.data._p[-1])
        ####################################################### sys identification processing time
        # Now process the image
        begin_time = time.time()
        self.proposed_steering_angles, self.uncertainty_list, self.selected_throttles = (self.steering_angle_planner.
                                                                                         optimization_step(current_image))
        selected_throttles = self.MPC_search(self.selected_throttles)
        print('search time: ', time.time() - begin_time)
        #self.searcher.step()
        #self.debug_(self.steering_angle_planner.searcher.status['pop_best'].values, self.searcher.status['pop_best'].values)
        return selected_throttles, self.proposed_steering_angles

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

    # image_subscriber = message_filters.Subscriber(imageTopicPath, Image)
    # imu_subscriber = message_filters.Subscriber(IMUTopicPath, Imu)
    # teensy_subscriber = message_filters.Subscriber('/teensy_serial', TeensySerial)
    # # sync the above-mentioned topics
    # max_delay = 1000
    # ts = message_filters.ApproximateTimeSynchronizer(
    #     [image_subscriber, teensy_subscriber, imu_subscriber], 30, max_delay)
    # ts.registerCallback(callbackSynced)

    image_subscriber = rospy.Subscriber(imageTopicPath, Image, callbackImage)
    imu_subscriber = rospy.Subscriber(IMUTopicPath, Imu, callbackImu)
    teensy_subscriber = rospy.Subscriber('/teensy_serial', TeensySerial, callbackTeensy)

    gps_subscriber = rospy.Subscriber(GPSTopicPath, NavSatFix, callbackGPS)

    command_publisher = rospy.Publisher('/jetson_command', RCCarCommand, queue_size=10)
    messange_to_get_published = RCCarCommand()
    messange_to_get_published.led_r = 100
    messange_to_get_published.led_g = 100
    messange_to_get_published.led_b = 100

    rate = rospy.Rate(20)
    ############################
    raw_image = None
    # for debugging purpose
    planner_command = None

    while (not flag_gps_data):
        time.sleep(0.5)
        print('Waiting for the GPS data...')
    print('GPS locked!')
    print('GPS location: ', gps_data)
    model_planning_horizon = 5
    mpc_planner_horizon = 5
    vision_planner_horizon = 5
    # our custom optimizer will use the following two lists to generate the solution population
    # There will be Gaussian PDFs whose means would be equal to these values in the lists with different variances
    steering_angle_actions_list = [-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8]
    throttle_actions_list = [0.2, 0.5, 0.8]
    starting_pop_size = 150
    # There are two stages, after evaluating the first population we will take the best ones to create the next generation
    second_phase_pop_size = 45
    main_planner = planner(model_planning_horizon, moh=vision_planner_horizon, mpcoh=mpc_planner_horizon,
                           gps_initial_data=gps_data,
                           vision_initial_pop_size=starting_pop_size,
                           vision_second_pop_size=second_phase_pop_size,
                           steering_angle_actions_list=steering_angle_actions_list,
                           throttle_actions_list=throttle_actions_list)
    # initialize the commands
    steering_angle = 0
    throttle = 0
    prev_time = 0
    steering_angle_list = []
    while not rospy.is_shutdown():
        begin_time = time.time()
        if raw_image is not None:

            current_image = input_preparation_practice(raw_image, augment=augment, debug_=False)
            if len(current_image)==0:
                continue

            given_measurements = [robot_orientation, gps_data, steering_angle, throttle]
            proposed_throttles, proposed_steering_angles = main_planner.plan(current_image, given_measurements)
            list_actions.append(proposed_throttles)
            throttle = decPlace(float(proposed_throttles[0].detach().cpu()))
            steering_angle = decPlace(float(proposed_steering_angles[0].detach().cpu()))
            # uncertainties.append(main_planner.steering_angle_planner.model_uncertainties)
            # recorded_rewards.append(main_planner.steering_angle_planner.temporal_rewards)
            # recorded_speeds.append(main_planner.speed)
            if teensy_data.ch_mode == -1:
                messange_to_get_published.steering = steering_angle
                messange_to_get_published.throttle = throttle
                command_publisher.publish(messange_to_get_published)

            planner_command = (throttle, steering_angle)
            print('commands (thr, str): ', throttle, steering_angle)
            steering_angle_list.append(steering_angle)
            count_sample +=1

        rate.sleep()
        prev_time = 0.6*prev_time + 0.4*( time.time() - begin_time)
        print('Avg time: ',prev_time)
# plt.plot(steering_angle_list)
# plt.show()
# input()
