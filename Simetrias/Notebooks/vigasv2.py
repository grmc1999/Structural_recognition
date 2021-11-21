import sys
import pathlib
pth=str(pathlib.Path().absolute())
sys.path.append(('\\').join(pth.split('\\')[:-1])+"\\Utils")
from Utilities import *
from MF import *

from Signatures import Signature
import os
import open3d as o3d
import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

def return_rot(vect, ang):
    cosA = np.cos(ang)
    sinA = np.sin(ang)
    oneMinusCosA = 1-cosA
    out = np.zeros((3, 3))          
    out[0, 0] = (cosA + vect[0] * vect[0] * oneMinusCosA)
    out[0, 1] = (vect[1] * vect[0] * oneMinusCosA + vect[2] * sinA)
    out[0, 2] = (vect[2] * vect[0] * oneMinusCosA - vect[1] * sinA)
    out[1, 0] = (vect[0] * vect[1] * oneMinusCosA - vect[2] * sinA)
    out[1, 1] = (cosA + vect[1] * vect[1] * oneMinusCosA)
    out[1, 2] = (vect[2] * vect[1] * oneMinusCosA + vect[0] * sinA)
    out[2, 0] = (vect[0] * vect[2] * oneMinusCosA + vect[1] * sinA)
    out[2, 1] = (vect[1] * vect[2] * oneMinusCosA - vect[0] * sinA)
    out[2, 2] = (cosA + vect[2] * vect[2] * oneMinusCosA)
    return out

def slices_(pcd,  scale = 1, N = 10):

    ev1,_=project_pcl_pca(pcd,
             voxel_size=0.02,
             nb=5,
             std_ratio=1.5,
             p_th=0.1,
             ig="Linearity",
             upper=True, 
             scaling_factor = scale
            ) 
    aux1 = np.cross(ev1, np.array([0.0, 0.0, 1.0]))
    aux1 = aux1/np.linalg.norm(aux1)
    aux2 = ev1[2]/np.linalg.norm(ev1)
    aux2 = np.arccos(aux2)
    rot = return_rot(aux1, aux2)
    rrot = return_rot(aux1, -aux2)
    pcd.rotate(rot)
    p = np.asarray(pcd.points)


    low_bound = np.min(p[:, 2])
    high_bound = np.max(p[:, 2])
    l = high_bound - low_bound
    N_SLICES = N
    nd = {i:0 for i in range(N_SLICES)}
    pd = {i:[] for i in range(N_SLICES)}

    for cont, i in enumerate(p):
        for j in range(N_SLICES):
            if i[2]>= low_bound+j*l/N_SLICES and i[2] < low_bound+(j+1)*l/N_SLICES:
                nd[j] += 1
                pd[j].append([p[cont, 0], p[cont, 1]])
                #p[cont, 2] = low_bound+j*l/N_SLICES

    for i in pd.keys():
        pd[i] = np.array(pd[i])
    return nd, pd, rrot

def return_biggest_lines(pd):
    N=len(pd)
    out = []
    for NN in range(N):
        a = pd[NN][:, 0]
        b = pd[NN][:, 1]
        x_shape = int(np.max(a) - np.min(a))
        y_shape = int(np.max(b) - np.min(b))
        m_shape = max(x_shape, y_shape)
        a = (a-np.min(a))*(400/m_shape)
        b = (b-np.min(b))*(400/m_shape)
        im = np.zeros((401, 401))

        indices = np.stack([a-1,b-1], axis =1).astype(int)
        im[indices[:,0], indices[:,1]] = 1

        im[indices[:,0], indices[:,1]] = 1

        image = im

        #lines = probabilistic_hough_line(image, threshold=45, line_length=10,
        #                                line_gap=40)
        lines = probabilistic_hough_line(image, threshold=15, line_length=10,
                                 line_gap=40, seed = 37)

        try:
            p0, p1 = lines[0]
            out.append(np.arctan((p0[1]-p1[1])/(p0[0]-p1[0])))
        except:
            pass
    out = np.array(out) 

    return out

def meta_beams(pcd, scale = 1, N=10):
    N = len(pcd.points)
    #if N>= 5000:
    #    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(N/200), std_ratio = 0.9)
    NT = len(pcd.points)
    if NT >= 4096:
        pcd = pcd.uniform_down_sample(int(NT/4096))
    bb = pcd.get_oriented_bounding_box()
    points = np.asarray(bb.get_box_points())
    d1 = np.linalg.norm(points[0]-points[2])
    d2 = np.linalg.norm(points[5]-points[2])
    RL = bb.R
    x, y, z = RL[0, 0], RL[1, 0], RL[2, 0]
    N = np.array([x, y, z])
    P = np.array(pts.get_center())
    s = []
    for i in pts.points:
        v = i-P
        dist = np.dot(v, N)
        #pp = i-dist*N
        s.append(dist)
    
    l = max(s) - min(s)
    base = P+N*min(s)
    nd, pd, R2 = slices_(pcd,scale =scale)
    an = return_biggest_lines(pd)
    n = []
    u = np.mean(an)
    s = np.std(an)
    if s >= np.pi/30:
        for i in an:
            if np.abs(i-u) <= 12*s/16:
                n.append(i)
        x_a = np.mean(np.array(n))
    else:
        x_a = u

    R1 = pcd.get_rotation_matrix_from_axis_angle([0., 0., (x_a+np.pi/15)])

    R = np.matmul(R1, R2)

    return R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1], R[0, 2], R[1, 2], R[2, 2], base, 0.9*d1, 0.9*d2, 0.1*d1, 0.1*d2, l, 0.01