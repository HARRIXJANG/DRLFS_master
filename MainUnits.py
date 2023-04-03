import numpy as np
import numpy.ma as ma
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SpherePoints:
    def __init__(self, interval_angle = 15):
        if 90 % interval_angle != 0:
            print("Please enter a number that is evenly divisible by 90!")
        self.interval_angle = interval_angle

    def Cartesian2Spherical(self, x, y, z):

        r = np.sqrt(np.power(x, 2)+np.power(y, 2)+np.power(z, 2))
        theta = np.arccos(z/r)
        phi = np.arctan(np.divide(y, x, out=y/(x+1e-6), where=x!=0))

        mask_x = x >= 0.0
        mask_y1 = y < 0.0
        mask_y2 = y >= 0.0

        mask_phi = ma.masked_array(phi, mask_x)
        mask_phi = ma.masked_array(mask_phi, mask_y1)
        mask_phi += math.pi
        mask_phi.mask = ma.nomask

        mask_phi = ma.masked_array(phi, mask_x)
        mask_phi = ma.masked_array(mask_phi, mask_y2)
        mask_phi -= math.pi
        mask_phi.mask = ma.nomask

        phi = np.array(mask_phi)

        theta_angle = 180.0 * theta / math.pi
        phi_angle = 180.0 * phi / math.pi
        return r, theta_angle, phi_angle

    def Spherical2Cartesian(self, r, theta, phi):
        r_theta = theta * math.pi / 180
        r_phi = phi * math.pi / 180
        x = r * np.sin(r_theta) * np.cos(r_phi)
        y = r * np.sin(r_theta) * np.sin(r_phi)
        z = r * np.cos(r_theta)
        return x, y, z

    def draw_point_cloud(self, x, y, z):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x, y, z, 'bo', color='green',markersize=2)
        plt.title('point_cloud')
        plt.show()

    def discretize_sphere(self):
        r = 1.0
        theta = np.arange(0, 181, self.interval_angle)
        phi = np.arange(0, 360, self.interval_angle)
        all_theta = theta
        for i in range(phi.shape[0]-1):
            all_theta = np.vstack((all_theta, theta))
        all_theta = all_theta.transpose(1,0).flatten()

        all_phi = phi
        for i in range(theta.shape[0]-1):
            all_phi = np.vstack((all_phi, phi))
        all_phi = all_phi.flatten()
        x, y, z = self.Spherical2Cartesian(r, all_theta, all_phi)
        all = x
        all = np.vstack((all, y))
        all = np.vstack((all, z))
        return all