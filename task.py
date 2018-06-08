import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.init_pose = init_pose
        self.init_velocities = init_velocities
        self.init_angle_velcities = init_angle_velocities

        self.state_size = self.action_repeat * 6
        self.action_low = 500
        self.action_high = 900
        self.action_size = 4

        self.init_pose = init_pose if init_pose is not None else np.array([25., 25., 120., 0, 0, 0])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.last_pos = np.array(self.init_pose[:3])


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #return 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        rotor_penalty = 1
        fall_penalty = 1



        #if np.max(self.rotor_speeds) - np.min(self.rotor_speeds) > 200:
        #    rotor = 0.5
        #print('Velocity')
        #print(self.sim.v)
        distance = np.sqrt(np.sum(np.square(self.sim.pose[:3] - self.target_pos)))
        prev_distance = np.sqrt(np.sum(np.square(self.last_pos[:3] - self.target_pos)))

        delta_d = prev_distance - distance
        #print(prev_distance, distance, delta_d)
        dd = self.sim.pose[2] - self.last_pos[2]
        self.last_pos = self.sim.pose
        #return dd
        return delta_d * rotor_penalty * fall_penalty


    def step(self, rotor_speeds):
        self.rotor_speeds = self.clip(rotor_speeds)

        """Uses action to obtain next state, reward, done."""
        #reward = 0
        #pose_all = []
        #for _ in range(self.action_repeat):
        done = self.sim.next_timestep(self.rotor_speeds) # update the sim pose and velocities
        reward = self.get_reward()
        #pose_all.append(self.sim.pose)
        #next_state = np.concatenate(pose_all)
        next_state = self.sim.pose

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat)
        #return state
        return self.sim.pose

    def new_target(self, target_pose):
        self.target_pos = target_pose
        print('Destination updated.')

    def clip(self, action):
        return np.clip(np.array(action), self.action_low, self.action_high)