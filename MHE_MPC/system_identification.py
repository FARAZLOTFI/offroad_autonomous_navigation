import numpy as np
import math
import sys
import os
import MHE_MPC.config as config
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

# Import do_mpc package:
import do_mpc

def GPS_deg2distance_XY(lon1,lon2,lat1,lat2):
    dx = (lon1 - lon2) * 40000 * np.cos((lat1 + lat2) * np.pi / 360) / 360
    dy = (lat1 - lat2) * 40000 / 360
    return dx*1000, dy*1000 # in meters

def GPS_deg2vel(lon1,lon2,lat1,lat2, time_taken=0.2):
    dx = ((lon1 - lon2) * 40000 * np.cos((lat1 + lat2) * np.pi / 360) / 360)*1000
    dy = ((lat1 - lat2) * 40000 / 360)*1000
    vel = np.sqrt(dx**2 + dy**2)/time_taken
    return vel, np.math.atan2(dy,dx)

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

class MHE_MPC():
    def __init__(self):
        self.model = do_mpc.model.Model('continuous') # either 'discrete' or 'continuous'
        self.model_initialization()

        self.data_counter = 0
        self.previous_file_number = None
        # Measurement configuration comes before state initialization as initialization is done based on the first meas
        self.offline_mode = True
        if self.offline_mode:
            self.topic_data_folder_path =  config.realworld_data_path + 'topics/'# TODO to put this in a config file
            self.topic_data_list = os.listdir(self.topic_data_folder_path)
            self.topic_data_list.sort()

        self.initial_GPS_lon = None
        self.initial_GPS_lat = None

        self.previous_GPS_lon = None
        self.previous_GPS_lat = None
        self.bearing = 0

        self.state_initialization()
        # MHE comes after all these as it needs the state initial value as the first guess
        self.mhe = do_mpc.estimator.MHE(self.model,
                                   ['pose_x', 'pose_y', 'global_orientation', 'velocity', 'Pitch', 'C1','Cm1', 'Cm2', 'Cr2', 'Cr0', 'mu_m'])
        self.MHE_initialization()

    def model_initialization(self):
        # OUR STATE VARIABLES ARE X Y SAI V Pitch
        X = self.model.set_variable(var_type='_x', var_name='pose_x', shape=(1, 1))
        Y = self.model.set_variable(var_type='_x', var_name='pose_y', shape=(1, 1))
        Sai = self.model.set_variable(var_type='_x', var_name='global_orientation', shape=(1, 1))
        V = self.model.set_variable(var_type='_x', var_name='velocity', shape=(1, 1))
        Pitch = self.model.set_variable(var_type='_x', var_name='Pitch', shape=(1, 1))

        # CONTROL INPUTS:
        # D: Dutycycle of the PWM signal which is identical to what we have as throttle
        # sigma: the steering angle-> for this we've steering throttle (-1~1) and it needs to be mapped
        D = self.model.set_variable(var_type='_u', var_name='throttle')
        sigma = self.model.set_variable(var_type='_u', var_name='steering_angle')

        # Setpoints given by the user e.g. to follow a trajectory
        X_set = self.model.set_variable(var_type='_u', var_name='des_pose_x')
        Y_set = self.model.set_variable(var_type='_u', var_name='des_pose_y')
        Sai_set = self.model.set_variable(var_type='_u', var_name='des_orientation')

        # Parameters:
        # C1 & C2 the two geometrical parameters that need to be set
        # Cm1 & Cm2 the two motor parameters that need to be identified using MHE
        C1 = self.model.set_variable('parameter', 'C1')  # geometrical param (-)
        C2 = 1/0.59 #self.model.set_variable('parameter', 'C2')  # geometrical param (1/m) -> measured as 59 cm
        Cm1 = self.model.set_variable('parameter', 'Cm1')  # motor param (m/s**2)
        Cm2 = self.model.set_variable('parameter', 'Cm2')  # motor param (1/s)
        Cr2 = self.model.set_variable('parameter', 'Cr2')  # second order friction param (1/m)
        Cr0 = self.model.set_variable('parameter', 'Cr0')  # zero order friction param (m/s**2)
        # added newly by Us to take into account the slope it is mass*mu(friction coefficient)
        mu_m = self.model.set_variable('parameter', 'mu_m')  # zero order friction param (m/s**2)
        g_ = 9.81
        # Measurements are noisy
        # State measurements
        # X & Y coming from GPS
        # Orientation coming from the IMU data
        # abs of V coming from GPS, yet the abs function is not defined; thus, we use V**2 instead
        X_measured = self.model.set_meas('X_pose_GPS', X, meas_noise=True)
        Y_measured = self.model.set_meas('Y_pose_GPS', Y, meas_noise=True)
        Sai_measured = self.model.set_meas('Heading_angle_IMU', Sai, meas_noise=True)
        vel_measured = self.model.set_meas('vel', V, meas_noise=True)

        slope_measured = self.model.set_meas('slope', Pitch, meas_noise=True)

        steering_angle_measured = self.model.set_meas('steering_angle_meas', sigma, meas_noise=False)
        throttle_measured = self.model.set_meas('throttle_meas', D, meas_noise=False)

        # model
        dX = V * np.cos(Sai + C1 * sigma)
        dY = V * np.sin(Sai + C1 * sigma)
        dSai = V * sigma * C2
        dV = (Cm1 - Cm2 * V ) * D - ((Cr2 * V ** 2 + Cr0) + (V * sigma)**2 * (C2 * C1 ** 2)) - \
             mu_m*g_*np.sin(Pitch)

        # The API does not support static augmentation, hence, I am using this simplified version
        dPitch = -0.000001*Pitch

        self.model.set_rhs('pose_x', dX, process_noise=True)
        self.model.set_rhs('pose_y', dY, process_noise=True)
        self.model.set_rhs('global_orientation', dSai, process_noise=True)
        self.model.set_rhs('velocity', dV, process_noise=True)
        # We consider Pitch as static
        self.model.set_rhs('Pitch', dPitch, process_noise=True)

        self.model.setup()

    def MHE_initialization(self):
        setup_mhe = {
            't_step': 0.2,
            'n_horizon': 5,
            'store_full_solution': True,
            'meas_from_data': True
        }
        self.mhe.set_param(**setup_mhe)

        num_of_measurements = 5
        num_of_states = 5
        num_of_parameters = 6
        P_v = np.eye(num_of_measurements)  # Covariance of the measurement noise
        P_x = np.eye(num_of_states)
        P_p = np.eye(num_of_parameters)
        P_w = np.eye(num_of_states)
        # P_w standing for the process noise

        self.mhe.set_default_objective(P_x, P_v, P_p, P_w)

        # bounds for the parameters
        lf = 0.3  # (m) center of gravity to the front wheel
        lr = 0.3  # (m) center of gravity to the front wheel
        self.mhe.bounds['lower', '_p_est', 'C1'] = 0.95 * lr / (lr + lf)
        self.mhe.bounds['upper', '_p_est', 'C1'] = 1.05 * lr / (lr + lf)

        # self.mhe.bounds['lower', '_p_est', 'C2'] = 0.9 * 1 / (lr + lf)
        # self.mhe.bounds['upper', '_p_est', 'C2'] = 1.1 * 1 / (lr + lf)

        self.mhe.bounds['lower', '_p_est', 'Cm1'] = 8
        self.mhe.bounds['upper', '_p_est', 'Cm1'] = 14 # -> 14 is observed frequently

        self.mhe.bounds['lower', '_p_est', 'Cm2'] = 1 # -> 1 is observed frequently
        self.mhe.bounds['upper', '_p_est', 'Cm2'] = 4

        self.mhe.bounds['lower', '_p_est', 'Cr2'] = 0.1
        self.mhe.bounds['upper', '_p_est', 'Cr2'] = 0.25
        #
        self.mhe.bounds['lower', '_p_est', 'Cr0'] = 0.6
        self.mhe.bounds['upper', '_p_est', 'Cr0'] = 0.9

        mass = 10
        self.mhe.bounds['lower', '_p_est', 'mu_m'] = 0.3*mass
        self.mhe.bounds['upper', '_p_est', 'mu_m'] = 0.6*mass
        # it has been identifiedV as 6 consistently

        self.mhe.setup()

        self.mhe.x0 = self.states
        # C1, Cm1&2, Cr2, Cr0, m*mu
        self.mhe.p_est0 = np.array([0.5, 12, 2.5, 0.15, 0.7, 4.0]).reshape(-1, 1)
        self.mhe.set_initial_guess()

    def state_initialization(self):
        # based on that we initiate our MHE model -> it needs to come from the first measurement
        lon_GPS, lat_GPS, yaw, pitch, steering_angle, throttle = self.measurement_update()
        # this we use as the origiVn to calculate the distance from this until the end
        self.initial_GPS_lon = lon_GPS
        self.initial_GPS_lat = lat_GPS
        self.previous_GPS_lon = lon_GPS
        self.previous_GPS_lat = lat_GPS

        self.initial_heading = yaw
        # we assume that the velocity is zero in the beginning
        self.states = np.array([0.0, 0.0, 0.0, 0.0, pitch]).reshape(-1, 1)

    def MHE(self):
        pass

    def measurement_update(self, offline_mode=True):

        if offline_mode:
            file_path = self.topic_data_folder_path + self.topic_data_list[self.data_counter]
            file_number = int(float(file_path[-8:-4]))
            if self.previous_file_number is not None:
                if file_number - self.previous_file_number == 1:  # TODO this should be used
                    flag_new_rosbag = False
                else:
                    flag_new_rosbag = True

            self.previous_file_number = file_number
            # the order of the data is: steering angle, throttle, w, x, y, z, Lon, lat,
            # then, timestamps for the image, depth, teensy, imu, and gps topics
            loaded_data = np.load(file_path)
            steering_angle, throttle, w, x, y, z, lon_GPS, lat_GPS = loaded_data[:8]
            self.data_counter += 1
        else:
            pass

        roll, pitch, yaw = euler_from_quaternion(x, y, z, w)
        pitch = -pitch
        if self.initial_GPS_lon is None:
            # note, the following is provided in degree! also we don't have vel as
            # at this point there is only one GPS point available
            return np.array([lon_GPS, lat_GPS, yaw, pitch, steering_angle*(-0.6), throttle])
        else: # we have to convert the degree to distance; the following gives us the distance in km
            dx, dy = GPS_deg2distance_XY(self.initial_GPS_lon, lon_GPS, self.initial_GPS_lat, lat_GPS)
            # dx and dy here are calculated nesbat be sharayet avalie na lahze ghabl k darvaghe ma lahze ghabl vasamoon moheme
            vel, dbearing = GPS_deg2vel(self.previous_GPS_lon, lon_GPS, self.previous_GPS_lat, lat_GPS)
            self.previous_GPS_lat = lat_GPS
            self.previous_GPS_lon = lon_GPS
            self.bearing = dbearing
            error = (yaw + 3.02) - dbearing

            return np.array([dx, dy, dbearing, vel, pitch, steering_angle * (-0.6), throttle])#, yaw


if __name__ == '__main__':
    system = MHE_MPC()
    angle_list = []
    # now the main loop
    for i in range(len(system.topic_data_list) - 1):
        y0 = system.measurement_update()
        #angle_list.append([y0[2],yaw,y0[5]])
        x0 = system.mhe.make_step(y0)

    np.save(config.realworld_data_path+ 'estimated_states.npy', system.mhe.data._x)
    np.save(config.realworld_data_path + 'estimated_parameters.npy', system.mhe.data._p)