import numpy as np
import car
#import random
import matplotlib.pyplot as plt
import time

import cvxopt
from cvxopt import matrix
from cvxopt import solvers
class PID:

    def __init__(self,P,I,D,dt,max_angle):

        self.Kp = P
        self.Kd = D
        self.Ki = I

        self.dt = dt

        self.sample_time = 0.00
        self.max_angle = max_angle

        self.clear()


    def clear(self):

        self.setPoint = 0.00

        self.Pterm = 0
        self.Iterm = 0
        self.Dterm = 0

        self.last_error = 0

        self.output = 0


    def update(self,setPoint,feedback_value):

        self.error = setPoint-feedback_value

        self.Pterm = self.Kp*self.error

        self.Iterm += self.error*self.dt

        self.delta_error = self.error - self.last_error

        self.Dterm = self.delta_error/self.dt

        self.last_error = self.error

        self.output = self.Pterm + self.Ki*self.Iterm + self.Dterm*self.Kd

        if self.output >= 0:
            self.output = min(self.output, self.max_angle)
        else:
            self.output = max(self.output, -self.max_angle)


        return  self.output


class Linear:
    def __init__(self,angle):

        self.angle = angle

    def update(self, setPoint, feedback_value):


        if setPoint+0.3 < feedback_value:
            self.output = -self.angle

        elif setPoint-0.3 > feedback_value:
            self.output = self.angle
        else:
            self.output = 0

        return  self.output


class lateral_control:

    def __init__(self):
        self.lr = 2
        self.lf = 2

    def quintic(self,s):

        polyVec = np.array([1, s, s**2, s**3, s**4, s**5])
        dpolyVec = np.array([0, 1, 2*s, 3*(s**2), 4*(s**3),5*(s**4)])
        ddpolyVec = np.array([0, 0, 2, 6*s, 12 * (s ** 2), 20 * (s ** 3)])


        return  polyVec, dpolyVec, ddpolyVec

    def H_matrix(self,s):

        H = np.array([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,4,12*s,24*(s**2),40*(s**3)],
                      [0,0,12*s,36*(s**2),72*(s**3),120*(s**4)],
                      [0,0,24*(s**2),72*(s**3),144*(s**4),240*(s**5)],
                      [0,0,40*(s**3),120*(s**4),240*(s**5),400*(s**6)]],dtype=np.double)

        return H

    def Aeq_matrix(self,s0,sF):

        Vec0,dVec0,ddVec0 = self.quintic(s0)
        VecF, dVecF, ddVecF = self.quintic(sF)

        Aeq = np.array([Vec0,dVec0,ddVec0,VecF,dVecF,ddVecF],dtype=np.double)

        return Aeq

    def beq_vec(self,y0,yf):

        beq = np.array([y0,0,0,yf,0,0],dtype=np.double)

        return beq


    def solve(self,y0,yF,s0,sF):

        self.Aeq = matrix(self.Aeq_matrix(s0,sF),tc='d')
        self.beq = matrix(self.beq_vec(y0,yF),tc='d')
        H0 = self.H_matrix(s0)
        HF = self.H_matrix(sF)
        self.H = matrix(HF-H0,tc='d')
        self.f = matrix(np.zeros((6,1)),tc='d')
        start = time.time()
        self.sol = solvers.qp(self.H,self.f,A=self.Aeq,b=self.beq, kktsolver='ldl', options={'kktreg':1e-9})
        end = time.time()

        self.params = np.array([self.sol['x'][0],self.sol['x'][1],self.sol['x'][2],self.sol['x'][3],self.sol['x'][4],self.sol['x'][5]])

        return self.params, end-start

    def steering(self,s):


        Vec, dVec, ddVec = self.quintic(s)
        func = np.dot(Vec,self.params)
        dfunc = np.dot(dVec, self.params)
        ddfunc = np.dot(ddVec, self.params)

        curvature = np.divide(ddfunc,(1+dfunc**2)**(np.divide(3,2)))
        delta = np.arctan(np.multiply(np.divide(self.lf,self.lr+1),np.tan(curvature)))

        return delta

    def function_output(self,s):
        Vec, dVec, ddVec = self.quintic(s)
        func = np.dot(Vec, self.params)
        dfunc = np.dot(dVec, self.params)
        ddfunc = np.dot(ddVec, self.params)

        return func, dfunc, ddfunc





























