import glob
import os
import sys
import time
import math
import weakref

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random
import numpy as np
import tensorflow as tf
import cv2
import glob

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Activation, Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D

class CarEnv():
    def __init__(self):
        self.latest = "./CNN_Train_Models/cp-0130.ckpt"
        self.front_camera = None
        self.IM_WIDTH = 640
        self.IM_HEIGHT = 480
        self.SHOW = False
        self.collistion_hist = []
        self.video = []

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        inp, out = self.model_base_64x3_CNN()
        self.model_2 = Model(inputs=inp, outputs=out)
        self.model_2.load_weights(self.latest)


    def model_base_64x3_CNN(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(self.IM_HEIGHT, self.IM_WIDTH,3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Dense(3, activation="linear"))

        return model.input, model.output

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape(self.IM_HEIGHT, self.IM_WIDTH, 4)
        i3 = i2[:, :, :3]
        self.video.append(i3)
        if self.SHOW:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def col_callback(self, colli):
        self.collistion_hist.append(colli)

    def bound(self, thrott, steer, brake):
        if(thrott > 1.0):
            thrott = 1.0
        if(thrott < 0.0):
            thrott = 0.0
        if(steer > 1.0):
            steer = 1.0
        if(steer < -1.0):
            steer = -1.0
        if(brake > 1.0):
            brake = 1.0
        if(brake < 0.0):
            brake = 0.0
        return thrott, steer, brake

    def main(self):
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)

        try:

            world = client.get_world()
            ego_vehicle = None
            ego_cam = None
            ego_col = None
            # --------------
            # Spawn ego vehicle
            # --------------
                    
            ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name','ego')
            print('\nEgo role_name is set')
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color',ego_color)
            print('\nEgo color is set')

            spawn_points = world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if 0 < number_of_spawn_points:
                random.shuffle(spawn_points)
                ego_transform = spawn_points[0]
                ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
                print('\nEgo is spawned')
            else: 
                logging.warning('Could not found any spawn points')


            # --------------
            # Add a RGB camera sensor to ego vehicle. 
            # --------------
            
            cam_bp = None
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x", f'{self.IM_WIDTH}')
            cam_bp.set_attribute("image_size_y", f'{self.IM_HEIGHT}')
            cam_bp.set_attribute("fov",str(105))
            cam_location = carla.Location(2,0,1)
            cam_rotation = carla.Rotation(0,180,0)
            cam_transform = carla.Transform(cam_location,cam_rotation)
            ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.SpringArm)
            ego_cam.listen(lambda image: self.process_img(image))

            # --------------
            # Add collision sensor to ego vehicle. 
            # --------------

            col_bp = world.get_blueprint_library().find('sensor.other.collision')
            col_location = carla.Location(0,0,0)
            col_rotation = carla.Rotation(0,0,0)
            col_transform = carla.Transform(col_location,col_rotation)
            ego_col = world.spawn_actor(col_bp,col_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
            ego_col.listen(lambda colli: self.col_callback(colli))

            time.sleep(4)
            while self.front_camera is None:
                time.sleep(0.01)

            while True:
                world_snapshot = world.wait_for_tick()

                out = self.model_2.predict(np.array(self.front_camera).reshape(-1, *self.front_camera.shape)/255)[0]

                thrott = out[0]
                steer = out[1]
	            brake =out[2]
	            thrott = float(thrott)
	            brake = float(brake)
	            steer = float(steer)

	            if(steer>-0.05 and steer<0.05):
	                steer=0.0

                vel = ego_vehicle.get_velocity()
                velocity = 3.6*math.sqrt((vel.x**2)+(vel.y**2)+(vel.z**2))
                if(velocity > 40):
                    thrott = 0.0

                thrott, steer, brake = self.bound(thrott, steer, brake)
                print([thrott, steer, brake])
                ego_vehicle.apply_control(carla.VehicleControl(throttle=thrott, steer=steer, brake=brake))

                if len(self.collistion_hist) != 0:
                    break

        finally:

            out = cv2.VideoWriter('run20.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (self.IM_WIDTH, self.IM_HEIGHT))
             
            for i in range(len(self.video)):
                out.write(self.video[i])
            out.release()

            if ego_vehicle is not None:
                ego_vehicle.destroy()
                if ego_cam is not None:
                    ego_cam.stop()
                    ego_cam.destroy()
                if ego_col is not None:
                    ego_col.stop()
                    ego_col.destroy()
            print('\nNothing to be done.')


if __name__ == '__main__':

    try:
        env = CarEnv()
        env.main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_replay.')