# State-observer-based-PIML

We introduce a State observer-based PIML for leader-following tracking control. Here, PIML based time-varying parameter estimation method is used. The system block diagram is explained as below.
![block diagram](https://github.com/sjpark000/State-observer-PIML/blob/main/block_diagram.jpg)

## 1) Error-state observer
The state observer is designed by obtained gain through "robust_observer_revised.m", which guarantees exponential observability. Its state is used as an input of PIML.
## 2) Error-state observer based PIML
![observer-based PIML](https://github.com/sjpark000/State-observer-PIML/blob/main/observer_based_PIML.jpg)
A state-observer is applied as a physical model to estimate the state of the time-varying uncertain model. Compared to using the original kinematic error equation directly, proposed method showed better learning performance since the training rate is dependent on dynamic characteristics. By "State_observer_PIML_training.py", trained PIML model is generated as "trained_PIML.pth".
## 3) Time-Varying Parameter Estimation
Using a trained PIML model, parameter estimation is performed as following. This process is contained in "LF_tracking_control.py".
![parameter_estimation](https://github.com/sjpark000/State-observer-PIML/blob/main/parameter_Case1_new.png)
## 4) Tracking control simulation
The estiamted parameter is utilized for designing the feed-forward term of the tracking controller, which is the leader robot's velocity. The trajectories of leader and following robot are shown below.
![tracking control](https://github.com/sjpark000/State-observer-PIML/blob/main/traj_proposed_controller.jpg)
## 5) Experimental result
To verify the control performance in real world, experiment is performed. The leader robot draws trajectory with given velocity, and the following robot receives leader robot's position to calculate positon error state, and time-varying parameter is estimated for every time-step. This process is performed in real-time to track the leader robot. The following is the video of the experiment.
