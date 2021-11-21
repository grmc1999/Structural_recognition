import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from scipy.spatial.transform import Rotation as R
from Signatures import Signature
from Utilities import *

class Transformation:
    def __init__(self,signatureA=None,signatureB=None,rigid=None,reflection=None,clustering_weigths=None):
        self.wiegth=clustering_weigths
        self.v_toPoint=np.vectorize(pyfunc=self.toPoint,otypes=[np.float],signature='()->(n)')
        if signatureA!=None:
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
            r=R.from_matrix(self.R)
            self.ea=r.as_rotvec()
        
            Ac=signatureA.pointCoordinates
            Bc=signatureB.pointCoordinates
        
            if reflection:
                Ac[2]=(-1)*Ac[2]

            self.t=Bc-self.s*np.matmul(self.R,Ac)

            if reflection:
                Ac[2]=(-1)*Ac[2]        
        
    def toPoint(self,Transformation):
        return np.array([Transformation.s*Transformation.wiegth[0],
                        Transformation.ea[0]*Transformation.wiegth[1],
                        Transformation.ea[1]*Transformation.wiegth[2],
                        Transformation.ea[2]*Transformation.wiegth[3],
                        Transformation.t[0]*Transformation.wiegth[4],
                        Transformation.t[1]*Transformation.wiegth[5],
                        Transformation.t[2]*Transformation.wiegth[6]])

