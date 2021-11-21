import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from Signatures import Signature
from Utilities import *

class Transformation:
    def __init__(self,signatureA,signatureB,rigid,reflection):
        self.origin_index=signatureA.point_index
        self.image_index=signatureB.point_index
        
        if rigid:
            self.s=1
        else:
            self.s=(signatureA.Kmin / signatureB.Kmin + signatureA.Kmax / signatureB.Kmax) / 2
        
        A=np.concatenate(((np.array([signatureA.minCurv]).T),
                          (np.array([signatureA.maxCurv]).T),
                          (np.array([signatureA.normal]).T)),axis=1)
        if reflection:
            A[:,2]=(-1)*A[:,2]
        B=np.concatenate(((np.array([signatureB.minCurv]).T),
                          (np.array([signatureB.maxCurv]).T),
                          (np.array([signatureB.normal]).T)),axis=1)
        
        self.R=np.matmul(B,np.transpose(A))
        
        Ac=signatureA.pointCoordinates
        Bc=signatureB.pointCoordinates
        
        if reflection:
            Ac[2]=(-1)*Ac[2]

        self.t=Bc-self.s*np.matmul(self.R,Ac)

        if reflection:
            Ac[2]=(-1)*Ac[2]        
        
    def toPoint(self):
        ea=rot2eul(self.R)
        return [self.s,ea[0],ea[1],ea[2],self.t[0],self.t[1],self.t[2]]
    
    #Vectorizar
    def toPoints(Transformations):
        Points=[]
        for t in Transformations:
            Points.append(t.toPoint())
        return np.array(Points)

