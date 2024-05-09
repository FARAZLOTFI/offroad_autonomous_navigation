# the main idea in this code is to further parallelize everything
# we have a list of steering angles
# we have a list of throttles
# instead of using MPC to iterate over the image-based model; we can evaluate all the possible throttles and
# then give the MPC a list of loss/uncertainty values; then the mpc can iterative over the kinematic model
# to find the best throttle given the corresponding loss term coming from the list given by the image-based planner.
import os
import random
from src.offroad_autonomous_navigation.metrics import Metrics
import torch
from src.offroad_autonomous_navigation.models.nn_model import PredictiveModelBadgr, LSTMSeqModel, TransformerSeqModel
from torchvision import transforms
import config
import numpy as np
import time
class CEM_optimizer():
    # for now consider batch and pop size equal to each other
    def __init__(self, steering_angle_actions_list = [-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8],
                 throttle_actions_list = [0.2, 0.5, 0.8],
                 model=None, batch_size=150, initial_pop_size=150, second_pop_size=45, planning_horizon=40, model_receding_horizon=40):
        self.planning_horizon = planning_horizon
        self.model_receding_horizon = model_receding_horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_space = [steering_angle_actions_list, throttle_actions_list]
        self.num_throttles = len(throttle_actions_list)
        self.predictive_model = model
        self.initial_pop_size = initial_pop_size
        self.second_pop_size = second_pop_size
        if batch_size>initial_pop_size:
            print('Batch size larger than the pop size; setting the batchsize equal to pop size...')
            batch_size = initial_pop_size
        self.batch_size = self.initial_pop_size
        begin_time = time.time()
        #self.initialize_search()
        self.scaler = 0.6
        self.metrics = Metrics(planning_horizon=planning_horizon, device=self.device)
        self.events_rewards = torch.tensor(np.array([-1, -1, -1, -1, -1, -1, 0.8, 1, 0.8]),
                                           dtype=torch.float32).cuda()
        self.desired_heading = torch.tensor(np.zeros([model_receding_horizon, 1]), dtype=torch.float32).cuda()

    def initialize_search(self, cluster_center_ref=None):
        # 2 because of throttle and steering angle; also we might have model which predicts for longer horizon but
        # we want to plan for shorter length of actions
        self.actions = torch.zeros(size=(self.model_receding_horizon, self.batch_size, 2)).cuda()
        self.steering_pop = []
        self.throttles_pop = []
        while len(self.steering_pop)< self.batch_size:
            if cluster_center_ref is None:
                cluster_center = random.choices(self.action_space[0],k=1)
            else:
                cluster_center = cluster_center_ref

            std_dev = 0.1 * np.random.rand() + 0.01  # at least 0.01 variance
            # for each throttle we would assume a distribution with that throttle as the mean to provide
            # the network with a distribution not constant values
            for i in range(len(self.action_space[1])):
                cluster_center_throttles = self.action_space[1][i]
                self.steering_pop.append(np.random.normal(cluster_center, std_dev, self.planning_horizon))
                self.throttles_pop.append(np.random.normal(cluster_center_throttles, std_dev, self.planning_horizon))
        self.throttles_pop = torch.tensor(self.throttles_pop,dtype=torch.float32).cuda()
        self.steering_pop = torch.tensor(self.steering_pop,dtype=torch.float32).cuda()

    def step(self,inputs, current_heading_angle):
        batch_inputs = inputs.repeat(self.batch_size, 1, 1, 1)
        print('batch size: ',batch_inputs.shape)
        part_loss = []
        uncertainties = []

        self.actions[:self.planning_horizon,:,0] = self.steering_pop.t()*self.scaler
        self.actions[:self.planning_horizon,:,1] = self.throttles_pop.t()

        inputs_ = [batch_inputs, self.actions]
        classification_output, pred_regression, epi_unc_classification, epi_unc_regressions = (
            self.metrics.calc_unc(self.predictive_model, inputs=inputs_))  # TODO BUG Potentially

        pred_classification, mix_dist = classification_output
        self.model_uncertainties = epi_unc_classification
        ##events, bearings = self.predictive_model.predict_events(self.image_embedding, self.actions)
        events = pred_classification
        loss = torch.tensor(np.zeros(self.batch_size), dtype=torch.float32).cuda()
        ### to go straight
        current_heading_angle = torch.tensor(current_heading_angle, dtype=torch.float32).cuda()
        # puting the variables on the GPU
        pred_classification = torch.tensor(pred_classification, dtype=torch.float32).cuda()
        pred_regression = torch.tensor(pred_regression, dtype=torch.float32).cuda()
        mix_dist = torch.tensor(mix_dist, dtype=torch.float32).cuda()

        for j, event in enumerate(events):
            # each event stands for the probability of that event to happen
            loss += self.events_rewards[event]
            # in either case of {tree,other_obstacles,human}, we don't care about the heading
            # TODO, i was doing a test on May 8 so I commented out the below line to ignore the heading
            #### loss += ((pred_regression[j] - self.desired_heading[j]) *
            ####          (torch.ones(1).cuda() - torch.max(mix_dist[j][:,:3],dim=1)[0]))
            # TODO May 8 to go straight
            loss += (0.5/(self.actions[j,:,0]**2 + 1) *
                     (torch.ones(1).cuda() - torch.max(mix_dist[j][:,:3],dim=1)[0]))
        # if self.prev_loss is None or self.prev_loss>loss:
        #     self.prev_loss = loss
        #     self.best_index = i

        part_loss.append(loss)
        uncertainties.append(epi_unc_classification)

        return torch.cat(part_loss, dim=0), torch.cat(uncertainties, dim=0)

    def optimization_step(self,image_tensor, current_heading_angle=0.0):
        self.batch_size = self.initial_pop_size
        self.initialize_search()
        loss, uncertainty = self.step(image_tensor, current_heading_angle)
        best_index = torch.argmax(loss)
        cluster_center = self.steering_pop[best_index].mean().cpu().detach().numpy()

        second_stage=True
        if second_stage:
            self.batch_size = self.second_pop_size
            #self.initial_pop_size = self.second_pop_size
            self.initialize_search(cluster_center_ref=cluster_center)
            loss, uncertainty = self.step(image_tensor, current_heading_angle)
            best_index = torch.argmax(loss)

        indexes = range(best_index - best_index % self.num_throttles,
                        best_index - best_index % self.num_throttles + self.num_throttles)
        uncertainty_list = []
        selected_throttles = []
        for index_ in indexes:
            uncertainty_list.append(uncertainty[:,index_])
            selected_throttles.append(self.throttles_pop[index_])

        # TODO check the selected throttles and make sure they cover the three different throttles
        return self.steering_pop[best_index], uncertainty_list, selected_throttles

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flag_gps_data = False

    BATCH_SIZE = 150 # must be dividable by the number of throttles
    planning_horizon = 10
    model_receding_horizon = 40
    steering_angle_actions_list = [-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8]
    throttle_actions_list = [0.2, 0.5, 0.8]
    initial_pop_size = BATCH_SIZE
    if not BATCH_SIZE%len(throttle_actions_list) ==0:
        print('Please change the BATCHSIZE value to be dividable by: ',len(throttle_actions_list))
        input('')

    # the model
    num_event_types = 9 + 1  # one for regression
    n_seq_model_layers = 4
    seq_elem_dim = 16
    action_dimension = 2
    # seq_encoder = LSTMSeqModel(n_seq_model_layers, seq_elem_dim)
    seq_encoder = TransformerSeqModel(n_seq_model_layers, seq_elem_dim)
    predictive_model = PredictiveModelBadgr(planning_horizon, num_event_types,
                                                 action_dimension, seq_encoder, n_seq_model_layers, device=device,
                                                 ensemble_size=3)
    predictive_model.cuda()

    # CHECKPOINT_PATH = config.weights_file
    #
    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    #predictive_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    predictive_model.eval()

    model_uncertainties = torch.zeros(planning_horizon).cuda()
    desired_heading = torch.tensor(np.zeros([planning_horizon, 1]), dtype=torch.float32).cuda()

    optimizer_ = CEM_optimizer(steering_angle_actions_list,
                               throttle_actions_list,
                               predictive_model,
                               BATCH_SIZE,
                               initial_pop_size,
                               planning_horizon=planning_horizon,
                               model_receding_horizon=model_receding_horizon)
    throttles =  torch.tensor(np.array(range(planning_horizon)), dtype=torch.float32).cuda()
    time_list = []
    for i in range(100):
        image_tensor = torch.randint(0, 255, size=(3, 72, 128), dtype=torch.float32).cuda()
        image_tensor = image_tensor.unsqueeze(0)

        begin_time = time.time()
        optimizer_.optimization_step(image_tensor)
        print('time taken to find the best solution: ', time.time() - begin_time)
        time_list.append(time.time() - begin_time)
    print('Average time: ',np.mean(np.array(time_list[10:])))