import numpy as np
import matplotlib.pyplot as plt
import car
import world
import agent
import lateral_agent




env = world.World(10,5,1000,30,0,1)
P = 1
I = 0
D = 0.5
max_angle = 2.5
dt = 0.05

done = False

lateral_controller = lateral_agent.lateral_control(dt)

goal_lane = (1 - 1) * env.road_width + env.road_width * 0.5
x_goal = 100
DELTA = []
POS_Y= []
POS_X = []
planned = []
y_acc = []
velocity = []
update = True
angle =[]
x_acc = []
y_acc_2 = []
for t in range(0,2000): #100s
    x_ego, y_ego, x_dot, y_dot ,v= env.get_ego()

    if goal_lane == y_ego:
        acc = env.dist_control(0)
        y_acc_2.append(0)
        steer = goal_lane
        y_acc.append(0)
        s_angle = 0



    else:
        steer,ydot,ydotdot = lateral_controller.function_output(x_ego)
        acc = env.dist_control(0)
        y_acc.append(lateral_controller.y_acceleration(x_ego, v))
        s_angle = lateral_controller.steering(x_ego)
        y_acc_2.append(ydotdot)

    if t == 100:
        goal_lane = (2 - 1) * env.road_width + env.road_width * 0.5
        _,_ = lateral_controller.solve(y_ego,goal_lane,x_ego,x_ego+50)
        update = False

    if t == 400:
        goal_lane = (3 - 1) * env.road_width + env.road_width * 0.5
        _,_ = lateral_controller.solve(y_ego,goal_lane,x_ego,x_ego+100)
        update = False

    if t == 800:
        goal_lane = (4 - 1) * env.road_width + env.road_width * 0.5
        _,_ = lateral_controller.solve(y_ego,goal_lane,x_ego,x_ego+50)
        update = False

    if t == 1200:
        goal_lane = (5 - 1) * env.road_width + env.road_width * 0.5
        _,_ = lateral_controller.solve(y_ego,goal_lane,x_ego,x_ego+50)
        update = False
    if t == 1600:
        goal_lane = (2 - 1) * env.road_width + env.road_width * 0.5
        _,_ = lateral_controller.solve(y_ego,goal_lane,x_ego,x_ego+100)
        update = False
    #acc = 0
    action = [acc,steer]
    env.step(action)

    #print("Velocity:", v, "X-Velocity:",x_dot,"Y-Velocity:",y_dot)
    angle.append(s_angle)
    POS_Y.append(y_ego)
    POS_X.append(x_ego)
    planned.append(steer)
    velocity.append(v)
    x_acc.append(acc)
    #env.render()




plt.figure(1)
plt.plot(POS_X,POS_Y)
plt.plot(POS_X,planned,'rx')
plt.title('Driven/Planned Path')
plt.xlabel('x-pos in [m]')
plt.ylabel('y-pos in [m]')
plt.show(block=False)


plt.figure(2)
ax1 = plt.subplot(3,1,1)
ax1.plot(POS_X,y_acc)
ax1.set_xlabel('x_pos in [m]')
ax1.set_ylabel('acc in [m/s^2]')
ax1.set_title('y-acceleration')
ax2 = plt.subplot(3,1,3)
ax2.plot(velocity)
ax2.set_xlabel('x_pps in [m]')
ax2.set_ylabel('velocity in [m/s]')
ax2.set_title('Velocity')
ax3 = plt.subplot(3,1,2)
ax3.plot(POS_X,x_acc)
ax3.set_xlabel('x_pos in [m]')
ax3.set_ylabel('acc in [m/s^2]')
ax3.set_title('x-acceleration')
plt.tight_layout()
plt.show(block=False)


plt.figure(3)
plt.plot(POS_X,np.rad2deg(angle))
plt.xlabel('x-pos in [m]')
plt.ylabel('steering angle in [degree]')
plt.title('Steering angle for path')
plt.show(block=False)


plt.figure(4)
ax1 = plt.subplot(2,1,1)
ax1.plot(POS_X,y_acc)
ax1.set_xlabel('x_pos in [m]')
ax1.set_ylabel('acc in [m/s^2]')
ax1.set_title('y-acceleration')
ax2 = plt.subplot(2,1,2)
ax2.plot(POS_X,y_acc_2)
ax2.set_xlabel('x_pos in [m]')
ax2.set_ylabel('acc in [m/s^2]')
ax2.set_title('y-acceleration from function')
plt.tight_layout()
plt.show()

