import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from Utilities import *
import functools
@vectorize
class Signature:
    
    def __init__(self,pointcloud=None,KDT=None,point_index=None):
        self.pointcloud=pointcloud
        self.point_index=point_index
        self.KDT = KDT
        if pointcloud!=None:
            self.pointCoordinates=np.asarray(self.pointcloud.points)[point_index]
    def set_signatures(self,kmin,kmax,knor,minCurv,maxCurv,normal):
        self.Kmin=kmin
        self.Kmax=kmax
        self.Knor=knor
        self.minCurv=minCurv
        self.maxCurv=maxCurv
        self.normal=normal
        self.ratio=self.Kmin/self.Kmax
        self.Sum=self.Kmin+self.Kmax+self.Knor
        self.Omnivariance=(self.Kmin*self.Kmax*self.Knor)**(1/3)
        #self.Eigenentropy=-(self.Kmin*math.log(self.Kmin)+self.Kmax*math.log(self.Kmax)+self.Knor*math.log(self.Knor))
        self.Anistropy=(self.Kmax-self.Knor)/self.Kmax
        self.Planarity=(self.Kmin-self.Knor)/self.Kmax
        self.Linearity=(self.Kmax-self.Kmin)/self.Kmax
        self.surfaceVariation=self.Knor/(self.Sum)
        self.sphericity=self.Knor/self.Kmax
        self.lineFeature=0
        self.K1_K3=self.Kmax-self.Kmin
        self.K1_K2=self.Kmax-self.Knor
        self.K1_K2_K3=self.Kmax-self.Knor-self.Kmin
        self.ratio_dis=(self.Knor/self.Kmin)-1

    def set_2D_signatures(self,kmin,kmax,knor,minCurv,maxCurv,normal):
        self.Kmax=kmax
        self.Knor=knor
        self.maxCurv=maxCurv
        self.normal=normal
        #self.ratio=self.Kmin/self.Kmax
        self.Sum=self.Kmax+self.Knor
        self.Omnivariance=(self.Kmin*self.Kmax*self.Knor)**(1/3)
        #self.Eigenentropy=-(self.Kmin*math.log(self.Kmin)+self.Kmax*math.log(self.Kmax)+self.Knor*math.log(self.Knor))
        self.Anistropy=(self.Kmax-self.Knor)/self.Kmax
        self.Planarity=(self.Kmin-self.Knor)/self.Kmax
        self.Linearity=(self.Kmax-self.Kmin)/self.Kmax
        self.surfaceVariation=self.Knor/(self.Sum)
        self.sphericity=self.Knor/self.Kmax
        self.lineFeature=0
        self.K1_K3=self.Kmax-self.Kmin
        self.K1_K2=self.Kmax-self.Knor
        self.K1_K2_K3=self.Kmax-self.Knor-self.Kmin
        self.ratio_dis=(self.Knor/self.Kmin)-1

    def build(self,NN_Criteria="RNN",NN=20,rad=0.6):
        Normals=np.zeros([3,1])
        MinCurDir=np.zeros([3,1])
        MaxCurDir=np.zeros([3,1])
        NormalVal=np.zeros([1,1])
        MinCurVal=np.zeros([1,1])
        MaxCurVal=np.zeros([1,1])
        [N,K1,K2]=getPrincipalDir(self.pointcloud,self.pointCoordinates,self.KDT,NN_Criteria,NN,rad)
        Normals=N[1]
        MinCurDir=K1[1]
        MaxCurDir=K2[1]
        NormalVal=N[0]
        MinCurVal=K1[0]
        MaxCurVal=K2[0]
        self.set_signatures(MinCurVal,MaxCurVal,NormalVal,MinCurDir,MaxCurDir,Normals)

    def dimension(self,rigid):
        if rigid:
            return 2
        else:
            return 1

    def is_umbilical(self,criteria,th):
        if criteria=="ratio":
            return abs(self.ratio)<th
        elif criteria=="sphericity":
            return abs(self.sphericity)>th

    def flatten(self,rigid):
        if rigid:
            return [self.Kmin,self.Kmax]
        else:
            return [self.Kmin/self.Kmax]

    @vectorize
    def v_build(self,signature,NN_Criteria="RNN",NN=20,rad=0.6):
        Normals=np.zeros([3,1])
        MinCurDir=np.zeros([3,1])
        MaxCurDir=np.zeros([3,1])
        NormalVal=np.zeros([1,1])
        MinCurVal=np.zeros([1,1])
        MaxCurVal=np.zeros([1,1])
        [N,K1,K2]=getPrincipalDir(signature.pointcloud,signature.pointCoordinates,signature.KDT,NN_Criteria,NN,rad)
        Normals=N[1]
        MinCurDir=K1[1]
        MaxCurDir=K2[1]
        NormalVal=N[0]
        MinCurVal=K1[0]
        MaxCurVal=K2[0]
        signature.set_signatures(MinCurVal,MaxCurVal,NormalVal,MinCurDir,MaxCurDir,Normals)

    @vectorize
    def v_flatten(self,signature,rigid):
        if rigid:
            return (signature.Kmin,signature.Kmax)
        else:
            return [signature.Kmin/signature.Kmax]
