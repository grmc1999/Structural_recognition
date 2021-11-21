import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from scipy import spatial
from Utilities import *
import random
from transformation import Transformation
from Signatures import Signature
from sklearn.cluster import MeanShift

def prune_points(signatures,th):
    pruned_points=[]
    for s in signatures:
        if s.is_umbilical(th):
            pruned_points.append(s)
    return pruned_points

def random_sample(signatures,percentage):
    rs=random.sample(signatures,int(len(signatures)*percentage))
    return rs

def build_pairing_kd_tree(pruned_points,rad=1e-02,rand_samp_percentage=0.1,rigid=True,only_reflections=True,NN=60):
    rs=random_sample(pruned_points,rand_samp_percentage)
    flats=Signature.flattena(rs,True)
    flatp=Signature.flattena(pruned_points,True)
    KDT=spatial.KDTree(flatp)
    query=KDT.query(flats,k=len(flats),eps=4,distance_upper_bound=rad)
    if only_reflections:
        z=1
    else:
        z=0
    for k in range(z+1):
        temp=[]
        for i in range(len(rs)):
            neighbors=query[1][i,:NN]
            sA=rs[i]
            for j in range(len(neighbors)):
                sB=pruned_points[neighbors[j]]
                if np.array_equal(sB.pointCoordinates,sA.pointCoordinates):
                    continue
                t=Transformation(sA,sB,rigid,bool(k))
                temp.append(t)
        if k==0:
            trans=np.array([temp])
        else:
            trans=np.concatenate((trans,np.array([temp])),axis=0).T
    return [trans,rs,KDT,query]


def run_clustering(Trans_Space,rigid=True,only_reflections=True,BW=500):
    if only_reflections:
        z=1
    else:
        z=0

    for k in range(z+1):
        points=Transformation.toPoints(Trans_Space[:,k])
        Cluster=MeanShift(bandwidth=BW,n_jobs=4)
        clusters=Cluster.fit_predict(points)
        if k==0:
            Cluster_trans=np.array([clusters])
        else:
            Cluster_trans=np.concatenate((Cluster_trans,np.array([clusters])),axis=0).T
    return [Cluster_trans,points]