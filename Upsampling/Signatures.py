import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from Utilities import *

class Signature:
    def __init__(self,pointcloud,point_index):
        self.pointcloud=pointcloud
        self.pointCoordinates=np.asarray(self.pointcloud.points)[point_index]
        self.point_index=np.arange(np.asarray(self.pointcloud.points).shape[0])[point_index]
    def set_signatures(self,kmin,kmax,minCurv,maxCurv,normal):
        self.Kmin=kmin
        self.Kmax=kmax
        self.minCurv=minCurv
        self.maxCurv=maxCurv
        self.normal=normal
        self.ratio=self.Kmin/self.Kmax
        
    def build_signatures(self,signatures,radius):
        [Normals,MinCurDir,MaxCurDir,MinCurVal,MaxCurVal]=pointCloudsPrincipalDirections(self.pointcloud,radius)
        for i in range(Normals.shape[0]):
            s=Signature(self.pointcloud,i)
            s.set_signatures(MinCurVal[i],MaxCurVal[i],MinCurDir[i],MaxCurDir[i],Normals[i])
            signatures.append(s)
    def dimension(self,rigid):
        if rigid:
            return 2
        else:
            return 1
    def flatten(self,rigid):
        if rigid:
            return [self.Kmin,self.Kmax]
        else:
            return [self.Kmin/self.Kmax]
    
    def flattena(signatures,rigid):
        flattened=[]
        for s in signatures:
            flattened.append(s.flatten(rigid))
        return flattened
    
    def is_umbilical(self,th):
        return abs(self.ratio)<th