import numpy as np


class Car:

    def __init__(self,x_init,y_init,velo_init,yaw_init,speed_limit,dt):

        # timestep
        self.dt = dt

        # Init position
        self.x = x_init
        self.y = y_init
        self.v = velo_init
        self.yaw_deg = yaw_init
        self.yaw = np.radians(self.yaw_deg)
        self.x_dot =0
        self.y_dot =0

        # Car parameters
        self.lr = 2 # Distance between reer wheel and center of gravity
        self.lf = 2 # Distance between front wheel and center of gravity
        self.a = 14
        self.b = 28
        self.max_v = speed_limit/3.6
        self.max_non_ego_v = self.max_v*0.7




    def motion(self,u1,u2): # u1 = acceleration, u2 = steering
        # Implementation of kinematic bicycle model
        # Slip Angle
        u2 = np.deg2rad(u2)
        self.beta = np.arctan(np.tan(u2)*np.divide(self.lr,self.lr + self.lf))

        # Velocities
        self.x_dot = self.v*np.cos(self.yaw+self.beta)
        self.y_dot = self.v*np.sin(self.yaw+self.beta)
        self.acc = u1
        self.yaw_dot = np.divide(self.v,self.lr)*np.sin(self.beta)

        # Movement
        self.x += self.x_dot*self.dt
        self.y += self.y_dot*self.dt
        self.yaw += self.yaw_dot*self.dt
        self.yaw_deg = np.degrees(self.yaw)
        self.yaw_dot_deg = np.degrees(self.yaw_dot)
        if self.v >= self.max_v and u1 > 0:
            self.v = self.max_v
        else:
            self.v += self.acc*self.dt

        if self.v < 0:
            self.v = 0

    def non_ego_motion(self,u1,u2): # u1 = acceleration, u2 = steering
        # Implementation of kinematic bicycle model
        # Slip Angle
        u2 = np.deg2rad(u2)
        self.beta = np.arctan(np.tan(u2)*np.divide(self.lr,self.lr + self.lf))

        # Velocities
        self.x_dot = self.v*np.cos(self.yaw+self.beta)
        self.y_dot = self.v*np.sin(self.yaw+self.beta)
        self.acc = u1
        self.yaw_dot = np.divide(self.v,self.lr)*np.sin(self.beta)

        # Movement
        self.x += self.x_dot*self.dt
        self.y += self.y_dot*self.dt
        self.yaw += self.yaw_dot*self.dt
        self.yaw_deg = np.degrees(self.yaw)
        self.yaw_dot_deg = np.degrees(self.yaw_dot)
        if self.v >= self.max_non_ego_v and u1 > 0:
            self.v = self.max_non_ego_v
        else:
            self.v += self.acc*self.dt

        if self.v < 0:
            self.v = 0

    def non_ego_motion_reversed(self,u1,u2): # u1 = acceleration, u2 = steering
        # Implementation of kinematic bicycle model
        # Slip Angle
        u2 = np.deg2rad(u2)
        self.beta = np.arctan(np.tan(u2) * np.divide(self.lr, self.lr + self.lf))

        # Velocities
        self.x_dot = self.v * np.cos(self.yaw + self.beta)
        self.y_dot = self.v * np.sin(self.yaw + self.beta)
        self.acc = u1
        self.yaw_dot = np.divide(self.v, self.lr) * np.sin(self.beta)

        # Movement
        self.x += self.x_dot * self.dt
        self.y += self.y_dot * self.dt
        self.yaw += self.yaw_dot * self.dt
        self.yaw_deg = np.degrees(self.yaw)
        self.yaw_dot_deg = np.degrees(self.yaw_dot)
        if self.v <= self.max_non_ego_v and u1 < 0:
            self.v = self.max_non_ego_v
        else:
            self.v += self.acc * self.dt

        if self.v > 0:
            self.v = 0

    def ego_motion(self,u1,u2):
        u2 = np.deg2rad(u2)

        self.beta = np.arctan(np.tan(u2) * np.divide(self.lr, self.lr + self.lf))

        # Velocities
        self.x_dot = self.v * np.cos(self.yaw + self.beta)
        self.y_dot = self.v * np.sin(self.yaw + self.beta)
        self.acc = u1
        self.yaw_dot = np.divide(self.v, self.lr) * np.sin(self.beta)

        # Movement
        self.x += self.x_dot * self.dt
        self.y += self.y_dot * self.dt
        self.yaw = u2
        self.yaw_deg = np.degrees(self.yaw)
        self.yaw_dot_deg = np.degrees(self.yaw_dot)



        if self.v >= self.max_v and u1 > 0:
            self.v = self.max_v
        else:
            self.v += self.acc * self.dt




















