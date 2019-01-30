import numpy as np
import car
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import lateral_agent
import copy
import time
import scipy.misc
from scipy import ndimage

import seaborn as sns
import collections


class World:

    def __init__(self,num_of_cars, lanes,length,speed_limit,init_pos,init_lane,ego_speed_init,dt,r_seed,x_range=10):

        self.n_lanes = lanes
        self.n_cars = num_of_cars
        self.x_range = x_range
        self.length = length
        self.speed_limit = speed_limit/3.6
        self.T = 1.5
        self.s0 = self.T * self.speed_limit
        self.delta = 4
        self.road_width = 4
        self.lane_init = (init_lane-1)*self.road_width + self.road_width*0.5
        self.dt = dt
        random.seed(r_seed)
        self.vehicle_list = []
        # Ego Car
        v_init = random.random() * (ego_speed_init/3.6)
        self.x_base = 0
        self.vehicle_list.append(car.Car(init_pos,self.lane_init,v_init,0,speed_limit,self.dt))
        self.dist_to_front = -1
        self.non_ego_limit = self.speed_limit * 0.4

        #POMDP Params
        self.x_view = 150
        self.y_view = 5
        self.resolution = 10 # 1m = resolution points
        self.observation_grid = np.zeros((self.x_view*2+1,self.y_view*2,2))

        # Car Dims
        self.car_x_dim = 4*self.resolution # 1x = 1m
        self.car_y_dim = 2*self.resolution # 2y = 1m

        ### RL Params
        self.done = False
        self.success = False
        self.reward = 0
        self.timestep = 0
        self.lane = self.lane_init
        self.lane_prev = self.lane_init
        self.x_goal = 3

        ### Occlusion stuff
        x_dim = (self.x_view * 2 + 1) * self.resolution
        y_dim = (self.y_view * 2 + 1) * self.resolution
        self.obs_body = np.zeros((x_dim, y_dim))
        self.grid = np.zeros((x_dim, y_dim))

        self.lateral_controller = lateral_agent.lateral_control(dt)

        ### Debug
        self.x_data = []
        self.y_data = []

        s = set()
        while len(s) < num_of_cars:
            s.add(np.random.choice(np.arange(0.3, 1, 0.05)) * self.length)

        alist = list(s)
        prev = 0
        for n in range(0, self.n_cars):

            y_random = random.randint(0, self.n_lanes - 1) * self.road_width + self.road_width * 0.5
            if prev == y_random and y_random == 2:
                y_random = 6
            elif (prev == y_random and y_random == 6):
                y_random = 2
            x_random = alist[n]
            v_random = random.random() * self.speed_limit * 0.4
            if y_random == 6:
                self.vehicle_list.append(car.Car(x_random, y_random, v_random * 1.3, 0, speed_limit, self.dt))
            elif (y_random == 2):
                self.vehicle_list.append(car.Car(x_random, y_random, v_random, 0, speed_limit, self.dt))
            prev = y_random


    def render(self):
        image_path_ego = get_sample_data('car-red.png')
        image_path_cars = get_sample_data('car-blue.png')
        local = np.argwhere(self.observation_grid[:,:,0]==1)


        plt.figure(1)
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ego = self.vehicle_list[0]
        ax1.plot([ego.x], [ego.y], 'ro')
        ax2.plot([ego.x], [ego.y], 'ro')
        # Car Pictures
        #imscatter([ego.x], [ego.y],image_path_ego,zoom=0.03,ax=ax1)
        #imscatter([ego.x], [ego.y], image_path_ego, zoom=0.03, ax=ax2)
        #ax2 = plt.step
        for lane in range(1,self.n_lanes+1):
            ax1.axhline((lane-1)*self.road_width + self.road_width*0.5, color="g", linestyle="-", linewidth=1)
            ax2.axhline((lane - 1) * self.road_width + self.road_width * 0.5, color="g", linestyle="-", linewidth=1)
            ax1.axvline(ego.x-self.x_view,color="b",linestyle="-",linewidth=1)
            ax1.axvline(ego.x+self.x_view, color="b", linestyle="-", linewidth=1)
        x_cars = []
        y_cars = []

        x_cars_o = []
        y_cars_o = []

        for n in range(1,self.n_cars+1):
            vehicle = self.vehicle_list[n]
            ax1.plot([vehicle.x], [vehicle.y], color = 'b',marker = 'x' )
            #ax2.plot([vehicle.x], [vehicle.y], color='b', marker='x')
            x_cars.append(vehicle.x)
            y_cars.append(vehicle.y)

        if(len(local)>0):
            for n in range(0,len(local)):
                x = int((local[n,0]-self.x_view)+ego.x)
                y = int((local[n,1]-self.y_view)+ego.y)
                if(y > 2):
                    print(y)
                x_cars_o.append(x)
                y_cars_o.append(y)
                ax2.plot(x,y, color='y', marker='o')

        #### First picture ####
        #imscatter(x_cars, y_cars, image_path_cars, zoom=0.03, ax=ax1)
        ax1.grid()
        ax1.set_ylim([1, (lane - 1) * self.road_width + self.road_width * 0.5 + 1])
        ax1.set_title("Global Map")
        ax1.set_ylabel("Y-Position in [m]")
        ax1.set_xlabel("X-Position in [m]")
        #imscatter([ego.x], [ego.y], image_path_ego, zoom=0.03, ax=ax1)

        #### Second Picture ###
        #ax2.set_ylim([0,(lane-1)*self.road_width + self.road_width*0.5+2])
        imscatter(x_cars_o, y_cars_o, image_path_cars, zoom=0.05, ax=ax2)
        imscatter([ego.x], [ego.y], image_path_ego, zoom=0.05, ax=ax2)
        ax2.set_ylim([0, (lane - 1) * self.road_width + self.road_width * 0.5 + 2])
        ax2.set_xlim([ego.x-self.x_view-50,ego.x+self.x_view+50])
        ax2.text(ego.x+10,ego.y+0.5, "Ego-velocity:"+str(round(ego.v,2))+ "\n" + "Distance: "+ str(round(self.dist_to_front,2)))
        ax2.set_title("Local Map")
        ax2.set_ylabel("Y-position in [m]")
        ax2.set_xlabel("X-position in [m]")
        plt.tight_layout()
        #img = plt.imread("images/road_2.jpg")
        #ax2.imshow(img)


        # Show
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(self.dt)
        plt.clf()

    def step(self,action):

        # Ego Action
        self.action = action
        self.decode_action()
        ego = self.vehicle_list[0]

        self.x_acc = self.dist_control(0)

        ego.motion(self.x_acc, self.steering)
        self.y_acc = ego.y_dot
        self.vehicle_list[0] = ego

        # Non-Ego Action
        for n in range(1, self.n_cars + 1):
            vehicle = self.vehicle_list[n]
            # acc, dist = self.IDM(n)
            acc = self.dist_control(n)
            vehicle.non_ego_motion(acc, 0)
            self.vehicle_list[n] = vehicle

        self.reward = self.reward_function()

        self.timestep += 1

        if self.timestep == 750:
            self.done = True

        self.lane_prev = self.lane
        vehicle_list_c = copy.deepcopy(self.vehicle_list)
        self.field_of_view()
        self.add_dimension()
        observation = copy.deepcopy(self.observation_grid)


        return vehicle_list_c, self.reward, self.done

    def IDM(self, id):

        # Get vehicles on same lane
        lane = self.vehicle_list[id].y
        x_pos = self.vehicle_list[id].x

        # Add Field of view
        vehicle_list_sec = [vehicle for vehicle in self.vehicle_list if vehicle.y <= lane + self.road_width*0.5 and vehicle.y >= lane - self.road_width*0.5 and vehicle != self.vehicle_list[id] and self.vehicle_list[id].x < vehicle.x]

        # Calculate distance to car in front
        if len(vehicle_list_sec) == 0:
            sa = self.s0*10
            v_delta =self.vehicle_list[id].v-self.speed_limit

        else:
            x_front = min(vehicle.x for vehicle in vehicle_list_sec)
            v_front = [vehicle.v for vehicle in vehicle_list_sec if vehicle.x ==x_front]
            sa = x_front - x_pos - 2*self.vehicle_list[id].lf
            v_delta = self.vehicle_list[id].v - v_front[0]

        # Implementation of IDM
        va = self.vehicle_list[id].v
        s_star = self.s0 + va*self.T + np.divide(va*v_delta,2*np.sqrt(self.vehicle_list[id].a*self.vehicle_list[id].b))
        acc = self.vehicle_list[id].a*(1 - (np.divide(va, self.speed_limit))**(self.delta) - (np.divide(s_star, sa))**2)

        return acc, sa

    def get_ego(self):

        ego = copy.deepcopy(self.vehicle_list[0])

        return ego.x, ego.y, ego.x_dot, ego.y_dot, ego.v


    def dist_control(self,id):

        alpha =0.5
        lane = self.vehicle_list[id].y
        x_pos = self.vehicle_list[id].x

        vehicle_list_sec = [vehicle for vehicle in self.vehicle_list if
                            vehicle.y <= lane + self.road_width * 0.5 and vehicle.y >= lane - self.road_width * 0.5 and vehicle !=
                            self.vehicle_list[id] and self.vehicle_list[id].x < vehicle.x]

        # Calculate distance to car in front
        if len(vehicle_list_sec) == 0:

            acc = alpha*(self.speed_limit - self.vehicle_list[id].v)
            if id == 0:
                self.dist_to_front = -1
        else:
            x_front = min(vehicle.x for vehicle in vehicle_list_sec)
            v_front = [vehicle.v for vehicle in vehicle_list_sec if vehicle.x == x_front]
            v_ego = self.vehicle_list[id].v
            dist = x_front - self.vehicle_list[id].x
            if id == 0:
                self.dist_to_front = dist

            acc = alpha*(v_front[0] - v_ego) + 0.25*(alpha**2)*(dist-self.s0)


        if acc >= 0:
            acc = min(self.vehicle_list[id].a,acc)
        else:
            acc = max(-self.vehicle_list[id].b,acc)

        if self.vehicle_list[id].v == self.speed_limit and acc > 0:
            acc = 0
        #acc += random.random()
        return acc

    def reward_function(self):
        self.reward = 0
        epsilon = 0.15
        self.reward -= (self.y_acc ** 2) * 0.12
        self.reward -= self.x_acc ** 2  # x_acc
        self.reward -= (self.speed_limit - self.vehicle_list[0].v) * 2
        self.lateral_dist = self.vehicle_list[0].y - 2
        self.reward += (1.375 * self.lateral_dist ** 2 - 6.25 * self.lateral_dist + 5)
        # if self.vehicle_list[0].y in {2,6,10,14,18,22}:
        #    self.reward += 1

        for lane in range(1, self.n_lanes + 1):
            if self.vehicle_list[0].y > (lane - 1) * self.road_width + self.road_width * 0.5 - epsilon and \
                    self.vehicle_list[0].y < (lane - 1) * self.road_width + self.road_width * 0.5 + epsilon:
                self.reward += 2
        #### Veloccity Delta ####
        non_ego_avg_v = 0
        for n in range(0, self.n_cars):
            non_ego_avg_v += self.vehicle_list[n + 1].v

        non_ego_avg_v = non_ego_avg_v / self.n_cars

        non_ego_delta_v = self.non_ego_limit - non_ego_avg_v

        ### Add Global part ###
        self.reward -= 0.1 * non_ego_delta_v ** 2

        vehicle_list_sec = [vehicle for vehicle in self.vehicle_list if self.vehicle_list[0].x < vehicle.x]

        if (len(vehicle_list_sec) == 0 and self.vehicle_list[0].y == (1 - 1) * self.road_width + self.road_width * 0.5):
            self.reward += 1000

            self.done = True
            self.success = True

        if (self.vehicle_list[0].y < 0 or self.vehicle_list[0].y > 2 + (
                self.n_lanes - 1) * self.road_width + self.road_width * 0.5):
            self.reward -= 1000
            self.done = True
            self.success = False
        #print("Reward: ",self.reward)

        return self.reward
    def field_of_view(self):

        self.number_of_frames = 4

        self.observation_grid = np.zeros((self.x_view*2+1,(self.y_view*2+1),2*self.number_of_frames))
        self.observation_temp = np.zeros((self.x_view*2+1,self.y_view*2+1,2*self.number_of_frames))

        for t in range(self.number_of_frames-1,-1,-1):

            if t> 0:
                self.observation_grid[:,:,2*t+1] = self.observation_grid[:,:,2*(t-1)+1]
                self.observation_grid[:, :, 2 * t ] = self.observation_grid[:, :, 2 * (t - 1)]


            elif (t == 0):
                # Maybe add ego position
                for car_no in range(1, self.n_cars + 1):

                    dist_x = self.vehicle_list[car_no].x - self.vehicle_list[0].x
                    dist_y = self.vehicle_list[car_no].y - self.vehicle_list[0].y
                    if abs(dist_x) < self.x_view and abs(dist_y) < self.y_view:

                        self.observation_grid[int(dist_x)+self.x_view,int(dist_y)+self.y_view,0] = 1
                        self.observation_grid[int(dist_x)+self.x_view, int(dist_y)+self.y_view, 1] = self.vehicle_list[0].v- self.vehicle_list[car_no].v

        observation = copy.deepcopy(self.observation_grid)

        return observation

    def add_dimension(self):
        x_dim = (self.x_view*2+1)*self.resolution
        y_dim = (self.y_view*2+1)*self.resolution

#      `self.obs_body = np.zeros((x_dim,y_dim,self.number_of_frames))
        self.obs_body = np.zeros((x_dim, y_dim))
        localization_global = np.argwhere(self.observation_grid[:,:,0]==1)
        localization_transformed = np.zeros(localization_global.shape)


        #ego_x = localization_global[0][0]
        #ego_y = localization_global[0][1]

        #### Transform into new resolution
        if len(localization_global) > 1:

            tic = time.time()
            for n in range(0,len(localization_global)):
                #print("x-before: ",localization_global[n][0])
                #print("y-before: ", localization_global[n][1])
                localization_transformed[n][0] = localization_global[n][0]*self.resolution
                localization_transformed[n][1] = localization_global[n][1]*self.resolution
                #print("x-before: ", localization_transformed[n][0])
                #print("y-before: ", localization_transformed[n][1])
            toc = time.time()
            #print("Time for loop: ",tic-toc)

            #### Add Body

            for n in range(0,len(localization_transformed)):
                #print("n: ",n)
                x_center = localization_transformed[n,0]
                y_center = localization_transformed[n,1]
                for x in range(0,self.car_x_dim):
                    #print("x: ", x)

                    x_grid_pos = int(x_center - self.car_x_dim * 0.5 + x)
                    if x_grid_pos < x_dim:

                        for y in range(0,self.car_y_dim):
                            y_grid_pos = int(y_center - self.car_y_dim * 0.5 + y)
                            #print("y: ", y)
                            if y_grid_pos < y_dim:
                                self.obs_body[x_grid_pos,y_grid_pos] = 1
                            else:
                                break
                    else:
                        break
            #unique_elements, counts_elements = np.unique(self.obs_body, return_counts=True)
            #print("Numbers in obs_grid: ",unique_elements, " Respective Amount: ",counts_elements,"-------\n\n")
            self.grid = copy.deepcopy(self.obs_body)
            #unique_elements, counts_elements = np.unique(self.grid, return_counts=True)
            #print("Numbers in grid before rays: ", unique_elements, " Respective Amount: ", counts_elements, "-------\n\n")
            self.rays()
            #unique_elements, counts_elements = np.unique(self.grid, return_counts=True)
            #print("Numbers in grid after rays: ", unique_elements, " Respective Amount: ", counts_elements,
                  #"-------\n\n")
            #print("Stop")


            #grid_90 = ndimage.rotate(self.grid,90)
            #plt.imshow(self.grid[:,:])
            #plt.imshow(grid_90[:, :])
            #plt.figure(10)
            #sns.heatmap(self.grid)
            #plt.tight_layout()
            #plt.colorbar()
            #plt.show(block=False)
            #plt.pause(self.dt*20)
            #plt.clf()
            #print("Stop")

            #### Implement Ray

            #### Use For whole grid


            ### Add dimension

        return self.obs_body

    def line_braham(self,x0,y0,x1,y1):
        dx = int(abs(x1-x0))
        dy = int(-abs(y1-y0))
        #self.grid = copy.deepcopy(self.obs_body)
        if(x0<x1):
            sx= 1
        else:
            sx= -1
        err = dx+dy

        if y0 < y1:
            sy = 1
        else:
            sy = -1

        test_number = 0
        #print("-----------Start-------------")
        #print("     Initial Values: Testnumber", test_number," X0: ", x0, "Y0: ", y0, "X1: ", x1, "Y1", y1)

        while(True):
            test_number+=1

            #self.grid[x0, y0] = 2
            if self.grid[x0,y0] == 1 or (x0 == x1 and y0 == y1):
                #print("     Final Values: Testnumber: ", test_number, "X0: ", x0, "Y0: ", y0, "X1: ", x1, "Y1", y1)
                self.x_data.append(x1)
                self.y_data.append(y1)
                break
            self.grid[x0, y0] = 2
            e2 = 2*err

            if e2 > dy:
                err = err + dy
                x0 = x0 + sx
            if e2 < dx:
                err = err + dx
                y0 = y0 + sy
        #print("-----------End-------------\n")

    def rays(self):


        x0 = int(self.x_view*self.resolution)
        y0 = int(self.y_view*self.resolution)
        x0_test = int(self.vehicle_list[0].x* self.resolution)
        y0_test = int(self.vehicle_list[0].y* self.resolution)
        count = 0
        for x1 in range(0,2*self.x_view*self.resolution):

            #print("#####Count: ",count)
            if x1 == 0 or x1 == 2*self.x_view*self.resolution-1:
                count += 1
                for y1 in range(0,2*self.y_view*self.resolution):
                    #print("x1up:",x1,"x0",x0)
                    #print("y1up:", y1,"y0",y0)
                    self.line_braham(x0,y0,x1,y1)
                    #self.grid[x1, y1] = 5


            else:
                count += 1
                #print("x1down:", x1, "x0", x0)
                #print("y1down:", y1, "y0", y0)
                #self.grid[x1,y1] = 5
                self.line_braham(x0,y0,x1,0)
                self.line_braham(x0,y0,x1,self.y_view*self.resolution*2-1)




    def return_acc(self):

        return self.x_acc,self.y_acc


    def decode_action(self):

        self.steering = self.action*0.5 -5
        #print("Action: ",self.action, " Lane selected: ",self.lane, "Distance Factor:", self.x_goal)

    def get_state(self):
        vehicle_list_c = copy.deepcopy(self.vehicle_list)


        return vehicle_list_c, self.reward,self.done




#### Additional plotting function for plotting

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists




