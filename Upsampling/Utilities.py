import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
import random


def find_isoplane(v,point,th=0.1):
    temp={}
    MD={}
    for i in range(len(v)):
        for j in range(len(v)-i-1):
            temp[(i,j+i+1)]=np.cross(v[i],v[i+j+1])/(np.linalg.norm(np.cross(v[i],v[i+j+1])))
    for i in temp:
        temp[i]=np.dot(temp[i],point)
    for i in temp:
        if abs(temp[i])<th:
            MD[i]=temp[i]
    return MD
def genSimFun(x,y,n):
    X=[]
    Y=[]
    for i in range(n+1):
        X.append(x**i)
        Y.append(y**i)
    X=sp.matrices.Matrix(X)
    Y=sp.matrices.Matrix(Y)
    XY=Y*X.transpose()
    return XY
def genFun(x,y,n):
    X=[]
    Y=[]
    for i in range(n+1):
        X.append(x**i)
        Y.append(y**i)
    X=np.array(X)
    X=np.transpose(X,axes=[1,0])
    X=X.reshape(X.shape[0],1,X.shape[1])
    Y=np.array(Y)
    Y=np.transpose(Y,axes=[1,0])
    Y=Y.reshape(Y.shape[0],1,Y.shape[1])
    Y=np.transpose(Y,axes=[0,2,1])
    XY=np.matmul(Y,X)
    XY=XY.reshape(x.shape[0],(n+1)**2)
    return XY
def transf(pnts,GR=3):
    x=pnts[:,0]
    y=pnts[:,1]
    z=pnts[:,2]
    XY=genFun(x,y,GR)
    Z=np.transpose(np.array([z]))
    return [XY,Z]
def fit(XY,Z):
    XYtZ=np.matmul(np.transpose(XY),Z)
    XYtXY=np.matmul(np.transpose(XY),XY)
    return np.matmul(np.linalg.inv(XYtXY),XYtZ)
def genSymFun(x,y,n=3):
    X=[]
    Y=[]
    for i in range(n+1):
        X.append(x**i)
        Y.append(y**i)
    X=sp.matrices.Matrix(X)
    Y=sp.matrices.Matrix(Y)
    XY=Y*X.transpose()
    XY=XY.reshape(1,(n+1)**2)
    return XY


def getPrincipalDir(pointcloud,point,KDT,radius=0.6):
    principalDirections={}
    [k, idx, _] = KDT.search_radius_vector_3d(point, radius)
    zp=np.asarray(pointcloud.points)[idx[:], :]
    zp=zp-np.mean(zp,axis=0)
    Cv=np.matmul(np.transpose(zp),zp)
    K,V=np.linalg.eig(Cv/k)
    if np.array_equal(K,np.array([0,0,0])):
        principalDirections=[(K[0],V[:,0]),(K[1],V[:,1]),(K[2],V[:,2])]
    else:
        if (np.cross(np.array(V[:,0]),pointcloud.normals[idx[0]])[0]<0):
            V[:,0]=(-1)*V[:,0]
            V[:,1]=(-1)*V[:,1]
        principalDirections={K[0]:V[:,0],K[1]:V[:,1],K[2]:V[:,2]}
        principalDirections=sorted(principalDirections.items())
    return principalDirections

def pointCloudsPrincipalDirections(pointCloud,radius=0.6):
    pt = o3d.geometry.KDTreeFlann(pointCloud)
    pointsc=np.asarray(pointCloud.points)
    r=range(pointsc.shape[0])
    a=pointsc.shape
    Normals=np.zeros(a)
    MinCurDir=np.zeros(a)
    MaxCurDir=np.zeros(a)
    MinCurVal=np.zeros(a[0])
    MaxCurVal=np.zeros(a[0])

    #Vectorizar------------------------------------------------------------------------
    for i in r:
        #[N,K1,K2]=getPrincipalDir(pointCloud,np.asarray(pointCloud.points)[i],pt,radius)
        [N,K1,K2]=getPrincipalDir(pointCloud,pointCloud.points[i],pt,radius)
        Normals[i]=N[1]
        MinCurDir[i]=K1[1]
        MaxCurDir[i]=K2[1]
        MinCurVal[i]=K1[0]
        MaxCurVal[i]=K2[0]
    return [Normals,MinCurDir,MaxCurDir,MinCurVal,MaxCurVal]

def principalDirections(pointCloud,point_index,radius=0.6):
    pt = o3d.geometry.KDTreeFlann(pointCloud)
    Normals=np.zeros([3,1])
    MinCurDir=np.zeros([3,1])
    MaxCurDir=np.zeros([3,1])
    MinCurVal=np.zeros([1,1])
    MaxCurVal=np.zeros([1,1])

    [N,K1,K2]=getPrincipalDir(pointCloud,point_index,pt,radius)
    Normals=N[1]
    MinCurDir=K1[1]
    MaxCurDir=K2[1]
    MinCurVal=K1[0]
    MaxCurVal=K2[0]
    return [Normals,MinCurDir,MaxCurDir,MinCurVal,MaxCurVal]


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


#Upsampling utilities

def upSampM1(pointcloud,KDT,radius):
    points=np.asarray(pointcloud.points)
    P=random.randint(0,len(points)-1)
    point=pointcloud.points[P]
    [k, idx, q] = pt.search_radius_vector_3d(point,radius)
    zp=np.asarray(dp.points)[idx[:], :]
    npo=(zp[0,:]+zp[-1,:])/2
    return npo

def upSampM2(pointcloud,KDT,r):
    points=np.asarray(pointcloud.points)
    P=random.randint(0,len(points)-1)
    point=pointcloud.points[P]
    [k, idx, q] = pt.search_radius_vector_3d(point,r)
    zp=np.asarray(dp.points)[idx[:], :]
    p1=zp[0,:]
    p2=zp[-1,:]
    d=np.linalg.norm(p1-p2)
    x=math.sqrt(r**2-(d/2)**2)
    pdp1=getPrincipalDir(dp,dp.points[P],pt,r)
    n1=pdp1[0][1]
    p12=p2-p1
    px=np.cross(p12,n1)
    px=px/(np.linalg.norm(px))
    p3=(p1+p2)/2+px*x
    return p3

def upsampling(pointcloud,KDT,r,N):
    npts=np.empty((1,3))
    for i in range(N):
        p=upSampM1(pointcloud,KDT,r)
        npts=np.concatenate((npts,p),axis=0)
    pointss=np.asarray(dp.points)
    pointss=np.concatenate((pointss,npts),axis=0)
    return pointss