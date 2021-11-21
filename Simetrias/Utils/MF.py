import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
from scipy import spatial
from Utilities import *
from Visualization_utilities import *
import random
from transformation import Transformation
from Signatures import Signature
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import copy
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as gridspec

def prune_points(signatures,criteria,th):
    pruned_points=[]
    for s in signatures:
        if s.is_umbilical(criteria,th):
            pruned_points.append(s)
    return np.array(pruned_points)

def pruneTransPoints(trans_s,
                            Rx_th=0,    #-
                            Ry_th=5,    #-
                            Rz_th_min=5,#-
                            Rz_th_max=0,#
                            Tz_th=-5,
                            Ty_th=0):
    tr=trans_s
    points=Transformation().v_toPoint(tr)
    rotx=points[:,1]
    roty=points[:,2]
    rotz=points[:,3]
    tray=points[:,4]
    traz=points[:,6]
    con1=np.abs(roty)<Ry_th
    con2=(np.abs(rotz)<Rz_th_min)+(np.abs(rotz)>Rz_th_max)
    con3=np.abs(rotx)>Rx_th
    con4=traz>Tz_th
    con5=np.abs(tray)>Ty_th
    return tr[np.where(con1*con2*con3*con4*con5)]

def random_sample(signatures,percentage):
    rs=random.sample(signatures.tolist(),int(len(signatures)*percentage))
    return np.array(rs)

def build_pairing_kd_tree(point_cloud,pruned_points,rad=1e-02,rand_samp_percentage=0.1,rigid=True,only_reflections=True,NN=60):
    diagonal_length=np.linalg.norm(point_cloud.get_max_bound()-point_cloud.get_min_bound())
    beta1=0.01
    beta2=1/((math.pi)**2)
    beta3=4/((diagonal_length)**2)
    wiegths=[beta1,beta2,beta2,beta2,beta3,beta3,beta3]
    rs=random_sample(pruned_points,rand_samp_percentage)
    if rigid==True:
        flats=Signature().v_flatten(np.array(rs),True)
        flats=np.concatenate((flats[0].reshape(-1,1),flats[1].reshape(-1,1)),axis=1)
        flatp=Signature().v_flatten(np.array(pruned_points),True)
        flatp=np.concatenate((flatp[0].reshape(-1,1),flatp[1].reshape(-1,1)),axis=1)
    else:
        flats=Signature().v_flatten(rs,rigid) #Rigid
        flatp=Signature().v_flatten(pruned_points,rigid) #Rigid
    KDT=spatial.KDTree(flatp)
    query=KDT.query(flats,k=len(flats),eps=1,distance_upper_bound=rad)
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
                t=Transformation(sA,sB,rigid,bool(k),wiegths)
                temp.append(t)
        if k==0:
            trans=np.array([temp])
        else:
            trans=np.concatenate((trans,np.array([temp])),axis=0).T
    return [trans,rs,KDT,query]


def run_clustering_Meanshift(Trans_Space,rigid=True,only_reflections=True,BW=500,min_bin_freq=1):
    if only_reflections:
        z=1
    else:
        z=0

    for k in range(z+1):
        points=Transformation.toPoints(Trans_Space[:,k])
        #Cluster=MeanShift(bandwidth=BW,n_jobs=4,cluster_all=False)
        #Cluster=MeanShift(bandwidth=BW,n_jobs=4,cluster_all=False,bin_seeding=True,min_bin_freq=500)
        Cluster=MeanShift(bandwidth=BW,n_jobs=4,cluster_all=False,bin_seeding=True,min_bin_freq=min_bin_freq)
        clusters=Cluster.fit_predict(points)
        if k==0:
            Cluster_trans=np.array([clusters])
        else:
            Cluster_trans=np.concatenate((Cluster_trans,np.array([clusters])),axis=0).T
    return [Cluster_trans,points]

def run_clustering_DBSCAN(eps_,min_samples_,Trans_Space,rigid=True,only_reflections=True):
    if only_reflections:
        z=1
    else:
        z=0

    for k in range(z+1):
        points=Transformation.toPoints(Trans_Space[:,k])
        Cluster=DBSCAN(eps=eps_,n_jobs=4,min_samples=min_samples_)
        clusters=Cluster.fit_predict(points)
        if k==0:
            Cluster_trans=np.array([clusters])
        else:
            Cluster_trans=np.concatenate((Cluster_trans,np.array([clusters])),axis=0).T
    return [Cluster_trans,points]

def run_clustering_OPTICS(min_samples,max_eps,xi,Trans_Space,rigid=True,only_reflections=True):
    if only_reflections:
        z=1
    else:
        z=0

    Clust_set=[]
    for k in range(z+1):
        points=Transformation().v_toPoint(Trans_Space[:,k])
        Cluster=OPTICS(min_samples=min_samples,max_eps=max_eps,xi=xi,n_jobs=4)
        clusters=Cluster.fit_predict(points)
        if k==0:
            Cluster_trans=np.array([clusters])
        else:
            Cluster_trans=np.concatenate((Cluster_trans,np.array([clusters])),axis=0).T
        Clust_set.append(Cluster)
    return [Cluster_trans,points,Clust_set]

def remove_discontinuities_by_curvature(pointcloud,signatures,KDT,curvature=0.5,radius=0.06):
    #signaturesp=copy.deepcopy(signatures)
    bc=prune_points(signatures,"ratio",curvature)
    #signatures.tolist().sort(key=lambda x: x.point_index, reverse=False)
    nps=signatures
    if bc!=[]:
        s=set()
        for sig in bc:
            index=sig.point_index
            [_, idx, _] = KDT.search_radius_vector_3d(pointcloud.points[index],radius)
            s=s.union(set(idx))
        idx=np.array(list(s))
        nps=np.array(signatures)
        nps=np.delete(nps,idx)
        #signatures=nps.tolist()
    return nps

#Proyección robusta
def project_pcl(pcl):
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=5,std_ratio=1.5)
    pcl=pcl.voxel_down_sample(voxel_size=0.02)
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

    indexes=np.arange(np.asarray(pcl.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(pcl)
    signatures=Signature(pcl,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(pcl.points)-np.mean(np.asarray(pcl.points),axis=0))/8,
                   NN=30)

    signaturesp=remove_discontinuities_by_curvature(pcl,signatures,pt,curvature=0.4,radius=0.5)

    pp=prune_points(signaturesp,"sphericity",0.5)
    points=np.vectorize(lambda x: x.pointCoordinates,signature="()->(j)")(pp)
    pts=o3d.geometry.PointCloud()
    pts.points=o3d.utility.Vector3dVector(points)

    obb=pts.get_oriented_bounding_box()
    dpp=copy.deepcopy(pcl)
    dpp.rotate(obb.R)
    #pa=np.asarray(dpp.points)
    #pa[:,0]=0
    #pts.points=o3d.utility.Vector3dVector(pa)
    return dpp

def project_pcl_(pcl):
    o3d.visualization.draw_geometries([pcl])
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=5,std_ratio=1.5)
    pcl=pcl.voxel_down_sample(voxel_size=0.02)
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    o3d.visualization.draw_geometries([pcl])

    indexes=np.arange(np.asarray(pcl.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(pcl)
    signatures=Signature(pcl,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(pcl.points)-np.mean(np.asarray(pcl.points),axis=0))/8,
                   NN=30)

    signaturesp=remove_discontinuities_by_curvature(pcl,signatures,pt,curvature=0.4,radius=0.5)
    graficarPropiedad(signaturesp,pcl,property="sphericity",frac=1)

    pp=prune_points(signaturesp,"sphericity",0.5)
    points=np.vectorize(lambda x: x.pointCoordinates,signature="()->(j)")(pp)
    pts=o3d.geometry.PointCloud()
    pts.points=o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pts])

    obb=pts.get_oriented_bounding_box()
    dpp=copy.deepcopy(pcl)
    dpp.rotate(obb.R)
    #pa=np.asarray(dpp.points)
    #pa[:,0]=0
    #pts.points=o3d.utility.Vector3dVector(pa)
    return dpp

def project_pcl_pca(pcl,voxel_size=0.08,nb=5,std_ratio=1.5,p_th=0.5,ig="sphericity",upper=True,scaling_factor=1):
    #inverser scaling
    pcl.scale(scaling_factor,center=pcl.get_center())
    #o3d.visualization.draw_geometries([pcl])
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=nb,std_ratio=std_ratio)
    pcl=pcl.voxel_down_sample(voxel_size=voxel_size)
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    #o3d.visualization.draw_geometries([pcl])

    indexes=np.arange(np.asarray(pcl.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(pcl)
    signatures=Signature(pcl,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(pcl.points)-np.mean(np.asarray(pcl.points),axis=0))/8,
                   NN=30)

    signaturesp=remove_discontinuities_by_curvature(pcl,signatures,pt,curvature=0.4,radius=0.5)
    #graficarPropiedad(signaturesp,pcl,property=ig,frac=1)

    #LESS THAN
    pp=np.vectorize(lambda signature,att: getattr(signature,att))(signaturesp,ig)
    if upper:
        pp=np.delete(signaturesp,pp>p_th)
    else:
        pp=np.delete(signaturesp,pp<p_th)
    points=np.vectorize(lambda x: x.pointCoordinates,signature="()->(j)")(pp)
    
    data_mean = np.mean(points, axis=0) #Calculate the average value of the column
    # Normalized 
    normalize_data = points - data_mean
    # SVD decomposition
    # Construct covariance matrix
    H = np.dot(normalize_data.T, normalize_data)
    # SVD decomposition
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)   # H = U S V
    # Reverse order
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvectors[:, 0],eigenvectors[:, 1]

def project_pcl_pca_2D(pcl,voxel_size=0.08,nb=5,std_ratio=1.5,p_th=0.5,ig="sphericity",upper=True,curvature=0.4,radius=0.5):
    #o3d.visualization.draw_geometries([pcl])
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=nb,std_ratio=std_ratio)
    pcl=pcl.voxel_down_sample(voxel_size=voxel_size)
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
    #o3d.visualization.draw_geometries([pcl])

    indexes=np.arange(np.asarray(pcl.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(pcl)
    signatures=Signature(pcl,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(pcl.points)-np.mean(np.asarray(pcl.points),axis=0))/8,
                   NN=30)

    signaturesp=remove_discontinuities_by_curvature(pcl,signatures,pt,curvature=curvature,radius=radius)
    #graficarPropiedad(signaturesp,pcl,property=ig,frac=1)

    #LESS THAN
    pp=np.vectorize(lambda signature,att: getattr(signature,att))(signaturesp,ig)
    if upper:
        pp=np.delete(signaturesp,pp>p_th)
    else:
        pp=np.delete(signaturesp,pp<p_th)
    points=np.vectorize(lambda x: x.pointCoordinates,signature="()->(j)")(pp)
    
    data_mean = np.mean(points, axis=0) #Calculate the average value of the column
    # Normalized 
    normalize_data = points - data_mean
    # SVD decomposition
    # Construct covariance matrix
    H = np.dot(normalize_data.T, normalize_data)
    # SVD decomposition
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)   # H = U S V
    # Reverse order
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvectors[:, 0],eigenvectors[:, 1]


def slices_(pcd, N = 10):

    ev1,ev2=project_pcl_pca(pcd,
             voxel_size=0.02,
             nb=5,
             std_ratio=1.5,
             p_th=0.1,
             ig="Linearity",
             upper=True
            ) 
    aux1 = np.cross(ev1, np.array([0.0, 0.0, 1.0]))
    aux1 = aux1/np.linalg.norm(aux1)
    aux2 = ev1[2]/np.linalg.norm(ev1)
    aux2 = np.arccos(aux2)
    rot = return_rot(aux1, aux2)
    pcd.rotate(rot)
    p = np.asarray(pcd.points)
    n_p = np.copy(p)


    low_bound = np.min(n_p[:, 2])
    high_bound = np.max(n_p[:, 2])
    l = high_bound - low_bound
    N_SLICES = N
    nd = {i:0 for i in range(N_SLICES)}
    pd = {i:[] for i in range(N_SLICES)}

    for cont, i in enumerate(n_p):
        for j in range(N_SLICES):
            if i[2]>= low_bound+j*l/N_SLICES and i[2] < low_bound+(j+1)*l/N_SLICES:
                nd[j] += 1
                pd[j].append([n_p[cont, 0], n_p[cont, 1]])
                n_p[cont, 2] = low_bound+j*l/N_SLICES
  
    return nd, pd