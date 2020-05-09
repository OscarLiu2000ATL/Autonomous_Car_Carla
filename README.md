# Autonomous_Car_Carla
The goal of this project is to train a Neural Network to drive in the driving simulation. What distinguishes my approach to that of similar projects is that my model can predict brake parameters in addition to steer and throttle parameters. The car would hit the brake depending on the obstacles and traffic light. This requires the Convolution Model to learn to extract not only road borders but also traffic lights' colors and road signs.


<div class="center" align="center">
  <img src="./1_5.gif"/>
</div>

## What is Carla
The main idea of Carla is to have the environment (server) and then agents (clients). This server/client architecture means that we can of course run both the server and client locally on the same machines, but we could also run the environement (server) on one machine and multiple clients on multiple other machines, which is pretty cool.

With Carla, we get a car (obviously), an environment to drive that car in, and then we have a bunch of sensors that we can place upon the car to emulate real-life self-driving car sensors. Things like LIDAR, cameras, accelerometers, and so on. [Python Programming Tutorials, pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/.]

## Train
### collect training data
python trains.py

### fit model
run.ipynb

## The Model
<div class="center" align="center">
  <img src="./2020-05-08 (2).png" width = "400"/>
</div>

## Training Graph
<div class="center" align="center">
  <img src="./2020-05-08 (3).png" width = "800"/>
</div>

## Final Results (Click To Open)
[![click](https://img.youtube.com/vi/hUkLMYN1Peo/0.jpg)](https://www.youtube.com/watch?v=hUkLMYN1Peo)
[![click_2](https://img.youtube.com/vi/1ad0i-BUCgY/0.jpg)](https://www.youtube.com/watch?v=1ad0i-BUCgY)

