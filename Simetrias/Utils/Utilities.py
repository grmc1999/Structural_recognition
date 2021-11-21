import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
import scipy as scp
from scipy import spatial
from sklearn.cluster import MeanShift
import random
import copy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import functools
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as gridspec





class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

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


def getPrincipalDir(pointcloud,point,KDT,NN_Criteria="RNN",NN=30,radius=0.6):
    principalDirections={}
    if NN_Criteria=="RNN":
        [k, idx, _] = KDT.search_radius_vector_3d(point, radius)
    elif NN_Criteria=="KNN":
        [k, idx, _] = KDT.search_knn_vector_3d(point, NN)
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

def pointCloudsPrincipalDirections(pointCloud,NN_Criteria="RNN",NN=30,radius=0.6):
    pt = o3d.geometry.KDTreeFlann(pointCloud)
    pointsc=np.asarray(pointCloud.points)
    r=range(pointsc.shape[0])
    a=pointsc.shape
    Normals=np.zeros(a)
    MinCurDir=np.zeros(a)
    MaxCurDir=np.zeros(a)
    NormalVal=np.zeros(a[0])
    MinCurVal=np.zeros(a[0])
    MaxCurVal=np.zeros(a[0])

    #Vectorizar------------------------------------------------------------------------
    for i in r:
        #[N,K1,K2]=getPrincipalDir(pointCloud,np.asarray(pointCloud.points)[i],pt,radius)
        [N,K1,K2]=getPrincipalDir(pointCloud,pointCloud.points[i],pt,NN_Criteria,NN,radius)
        Normals[i]=N[1]
        MinCurDir[i]=K1[1]
        MaxCurDir[i]=K2[1]
        NormalVal[i]=N[0]
        MinCurVal[i]=K1[0]
        MaxCurVal[i]=K2[0]
    return [Normals,MinCurDir,MaxCurDir,NormalVal,MinCurVal,MaxCurVal]

def principalDirections(pointCloud,point_index,NN_Criteria="RNN",NN=30,radius=0.6):
    pt = o3d.geometry.KDTreeFlann(pointCloud)
    Normals=np.zeros([3,1])
    MinCurDir=np.zeros([3,1])
    MaxCurDir=np.zeros([3,1])
    NormalVal=np.zeros([1,1])
    MinCurVal=np.zeros([1,1])
    MaxCurVal=np.zeros([1,1])

    [N,K1,K2]=getPrincipalDir(pointCloud,point_index,pt,NN_Criteria,NN,radius)
    Normals=N[1]
    MinCurDir=K1[1]
    MaxCurDir=K2[1]
    NormalVal=N[0]
    MinCurVal=K1[0]
    MaxCurVal=K2[0]
    return [Normals,MinCurDir,MaxCurDir,NormalVal,MinCurVal,MaxCurVal]


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def CalculateLineFeature(pointcloud,signatures,signature,KDT,alp=0.2,rad=0.06):
    index=signature.point_index
    [k, idx, _] = KDT.search_radius_vector_3d(pointcloud.points[index], rad)
    Cn=[]
    Cf=[]
    signatures.sort(key=lambda x: x.point_index, reverse=False)
    for i in idx[:int(len(idx)*alp)]:
        sig=signatures[i]
        Cn.append(sig.surfaceVariation)
    for i in idx[int(len(idx)*alp):]:
        sig=signatures[i]
        Cf.append(sig.surfaceVariation)
    Cn=np.array(Cn).mean()
    Cf=np.array(Cf).mean()
    return Cn/Cf

def get_cluster_transformation_points(Clusters,Cluster_id,trans_space):
    return trans_space[np.where(Clusters==Cluster_id)]

def get_signatures_from_transformation(Geometry_signatures,transformation_points):
    new_pc_ind=np.vectorize(pyfunc=(lambda x: np.array([x.origin_index,x.image_index])),
                            signature='()->(n)')(transformation_points)
    new_pc_ind=np.unique(new_pc_ind.reshape(1,-1))
    signatures=Geometry_signatures.tolist()
    signatures.sort(key=lambda x: getattr(x,"point_index"), reverse=False)
    signatures=np.array(signatures)
    return signatures[new_pc_ind]

get_cluster_points_index_from_signatures=np.vectorize(pyfunc=(lambda x:np.array([x.point_index])))

v_get_NN=np.vectorize(pyfunc=(lambda cpi,NN,ptc,KDT:np.asarray(np.array(KDT.search_knn_vector_3d(np.asarray(ptc.points)[cpi].reshape(-1,1),NN))[1])),
                      signature='(n),(),(),()->(k)')

def get_cluster_NN_points_index_from_signatures(cluster_signatures,pointcloud,KDT,NN):
    cluster_points_index=get_cluster_points_index_from_signatures(cluster_signatures)
    cluster_NN_point_index=v_get_NN(cluster_points_index.reshape(-1,1),NN,pointcloud,KDT).reshape(1,-1)
    return np.unique(cluster_NN_point_index).reshape(1,-1)

def build_pointcloud_simetrie(pointcloud,cluster_NN_point_index):
    Cluster_coordinates=np.asarray(pointcloud.points)[cluster_NN_point_index]
    Cluster_simetrie= o3d.geometry.PointCloud()
    Cluster_simetrie.points = o3d.utility.Vector3dVector(Cluster_coordinates[0])
    return Cluster_simetrie

def get_SSpace1(Space,pointCloud):
    diagonal_length=np.linalg.norm(pointCloud.get_max_bound()-pointCloud.get_min_bound())

    thKmin=Space[:,1]
    thKmax=Space[:,2]
    thKnor=Space[:,3]
    trKmin=Space[:,4]/(4/((diagonal_length)**2))
    trKmax=Space[:,5]/(4/((diagonal_length)**2))
    trKnor=Space[:,6]/(4/((diagonal_length)**2))

    Tp=np.linalg.norm(np.concatenate((trKnor.reshape(-1,1),trKmax.reshape(-1,1)),axis=1),axis=1) #primer
    Den=2*np.abs(np.sin(thKmin*(math.pi**2)/2))
    Den[np.where(Den>100000000000)]=100000000000
    Den[np.where(Den==0)]=0.0000000001
    Ra=Tp/(Den)
    base=10
    exponente=np.abs(thKnor.reshape(-1,1)*thKmax.reshape(-1,1)*(math.pi**4))
    #K=np.power(base,exponente)-1
    K=exponente
    Sub_space_1=np.concatenate((Ra.reshape(-1,1)*(4/((diagonal_length)**2)),thKmin.reshape(-1,1)/(math.pi**2),K),axis=1)
    return Sub_space_1

def get_SSpace2(Space,pointCloud):
    SS1=get_SSpace1(Space,pointCloud)
    SS2=SS1[:,(0,2)]
    return SS2

def Geometry_load(path="D:\\Documentos\\INNOVATE\\lib\\symmetry_detection_python\\Lineas 01_10.pts",
                    visualization=True,
                    voxel_down_sample=0.02,
                    geometry_type="pointCloud"):
    if geometry_type=="pointCloud":
        dp=o3d.io.read_point_cloud(path)
        dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
        #Visualización
        if visualization==True:
            o3d.visualization.draw_geometries([dp],point_show_normal=False)
        cl, ind = dp.remove_statistical_outlier(nb_neighbors=5,std_ratio=1.5)
        #Visualizacion
        if visualization==True:
            display_inlier_outlier(dp, ind)
        dp=cl
        dp=dp.voxel_down_sample(voxel_size=voxel_down_sample)
        dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
        #Visualizacion
        if visualization==True:
            o3d.visualization.draw_geometries([dp])
    elif geometry_type=="3DObject":
        mesh = o3d.io.read_triangle_mesh(path)
        dp = mesh.sample_points_uniformly(number_of_points=19500)
        dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
        dp=dp.voxel_down_sample(voxel_size=voxel_down_sample)
        #Visualizacion
        if visualization==True:
            o3d.visualization.draw_geometries([dp])
    return dp

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