import open3d as o3d
import numpy as np
import sympy as sp
import math
import scipy as scp
import random
import copy
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from transformation import Transformation
from Signatures import Signature
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def graficarPropiedad(signatures,dp,property="sphericity",frac=0.07,reverse=True):
    sig=signatures.tolist()
    sig.sort(key=lambda x: getattr(x,property), reverse=reverse)
    p=1/(len(sig))
    dp.paint_uniform_color([0.8, 0.8, 0.8])
    z=1
    for i in range(int(len(sig)*frac)):
        index=sig[i].point_index
        np.asarray(dp.colors)[index, :] = [0,0+p*i,1-p*i]
    o3d.visualization.draw_geometries([dp])

def densityPlot(kde_model,points):
    x=np.linspace(np.min(points[:,0])*1.8,np.max(points[:,0])*1.2,200)
    y=np.linspace(np.min(points[:,1])*1.8,np.max(points[:,1])*1.2,200)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.flatten(), yy.flatten()))
    log_densidad_pred = kde_model.score_samples(grid)
    densidad_pred = np.exp(log_densidad_pred)
    plt.style.use('default')
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, densidad_pred.reshape(xx.shape), cmap='viridis')
    ax.set_xlabel('intervalo (waiting)')
    ax.set_ylabel('duración (duration)')
    ax.set_zlabel('densidad')
    ax.set_title('Superficie 3D densidad')
    plt.show()
    plt.style.use('ggplot')

def plot_Clusters(z,trans_s,Clusters,colors,n_components):
    if n_components==2:
        embedding= MDS(n_components=2,n_jobs=4)
        points0=Transformation.toPoints(trans_s[:,z])
        points0 = embedding.fit_transform(points0)
        plot0=np.concatenate((np.zeros((points0.shape[0],1)),points0),axis=1)
        trans_g=o3d.geometry.PointCloud()
        trans_g.points = o3d.utility.Vector3dVector(plot0)
        trans_g.paint_uniform_color([0.8, 0.8, 0.8])
        for i in range(len(colors)):
            ind=np.where(Clusters[:,z]==i)
            np.asarray(trans_g.colors)[ind,:]=colors[i]
    elif n_components==3:
        embedding= MDS(n_components=3,n_jobs=4)
        points0=Transformation().v_toPoint(trans_s[:,z])
        points0 = embedding.fit_transform(points0)
        trans_g=o3d.geometry.PointCloud()
        trans_g.points = o3d.utility.Vector3dVector(points0)
        trans_g.paint_uniform_color([0.8, 0.8, 0.8])
        for i in range(len(colors)):
            ind=np.where(Clusters[:,z]==i)
            np.asarray(trans_g.colors)[ind,:]=colors[i]
    o3d.visualization.draw_geometries([trans_g])

def graf_cluster(ini,fin,rad,z,cluster,transformation_space,point_cloud,KDT):
    #Cluster_trans=MSCluster_trans
    p=1/(fin-ini)
    point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    for i in range(ini,fin):
        #ind=np.where(cluster[:,z]==i)
        ind=np.where(cluster==i)
        #trans_space_cluster=transformation_space[ind,z]
        trans_space_cluster=transformation_space[ind]
        for j in trans_space_cluster:
        #                                           R    G     B
            [k, idx, _] = KDT.search_radius_vector_3d(point_cloud.points[j.image_index], rad)
            np.asarray(point_cloud.colors)[idx[:], :] = [0,1-p*(i-ini),0+p*(i-ini)]
            [k, idx, _] = KDT.search_radius_vector_3d(point_cloud.points[j.origin_index], rad)
            np.asarray(point_cloud.colors)[idx[:], :] = [0,1-p*(i-ini),0+p*(i-ini)]
    o3d.visualization.draw_geometries([point_cloud])

def reachability_plot(Clust_set,trans_s,z=1):
    plt.figure(figsize=(30, 21))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, :])

    space=np.arange(len(trans_s[:,0]))
    reachability=Clust_set[0].reachability_[Clust_set[0].ordering_]
    labels=Clust_set[0].labels_[Clust_set[0].ordering_]

    colors = ['g.', 'r.', 'b.', 'y.', 'c.','m.','y.','g.', 'r.', 'b.', 'y.', 'c.','m.','y.']
    for klass, color in zip(range(0, 14), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    #ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    #ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    if z==1:
        space=np.arange(len(trans_s[:,1]))
        reachability=Clust_set[1].reachability_[Clust_set[1].ordering_]
        labels=Clust_set[1].labels_[Clust_set[1].ordering_]
    
        for klass, color in zip(range(0, 14), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            ax2.plot(Xk, Rk, color, alpha=0.3)
        ax2.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
        #ax2.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
        #ax2.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
        ax2.set_ylabel('Reachability (epsilon distance)')
        ax2.set_title('Reachability Plot')
    plt.show()



def display_trans_prunning(original_trans_space,pruned_trans_space):
    original_trans_space=Transformation().v_toPoint(original_trans_space)
    pruned_trans_space=Transformation().v_toPoint(pruned_trans_space)

    plt.figure(figsize=(30, 10))
    G = gridspec.GridSpec(6, 6)

    plt.subplot(G[1, 0]).hist(pruned_trans_space[:,1], bins='auto')
    plt.subplot(G[1, 1]).hist(pruned_trans_space[:,2], bins='auto')
    plt.subplot(G[1, 2]).hist(pruned_trans_space[:,3], bins='auto')
    plt.subplot(G[1, 3]).hist(pruned_trans_space[:,4], bins='auto')
    plt.subplot(G[1, 4]).hist(pruned_trans_space[:,5], bins='auto')
    plt.subplot(G[1, 5]).hist(pruned_trans_space[:,6], bins='auto')

    plt.subplot(G[0, 0]).hist(original_trans_space[:,1], bins='auto')
    plt.subplot(G[0, 1]).hist(original_trans_space[:,2], bins='auto')
    plt.subplot(G[0, 2]).hist(original_trans_space[:,3], bins='auto')
    plt.subplot(G[0, 3]).hist(original_trans_space[:,4], bins='auto')
    plt.subplot(G[0, 4]).hist(original_trans_space[:,5], bins='auto')
    plt.subplot(G[0, 5]).hist(original_trans_space[:,6], bins='auto')

    plt.subplot(G[0, 0]).set_title("Rx")
    plt.subplot(G[0, 1]).set_title("Ry")
    plt.subplot(G[0, 2]).set_title("Rz")
    plt.subplot(G[0, 3]).set_title("Tx")
    plt.subplot(G[0, 4]).set_title("Ty")
    plt.subplot(G[0, 5]).set_title("Tz")

    plt.show()

def display_sub_space(Sub_space):
    ax = plt.axes(projection='3d')
    ax.scatter3D(Sub_space[:,0], Sub_space[:,1], Sub_space[:,2])
    ax.set_xlabel('R')
    ax.set_ylabel('ThN')
    ax.set_zlabel('ThKmax')
    ax.set_xlim3d(0, 20)
    ax.set_zlim3d(0, 10)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(Sub_space[:,0],Sub_space[:,1],'bo')
    plt.axis([0,100,np.min(Sub_space[:,1]),np.max(Sub_space[:,1])])
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(Sub_space[:,0],Sub_space[:,2],'bo')
    plt.axis([0, 100, 0, 0.2])
    plt.show()

def display_cluster(clusters,Sub_space_2):
    colors=['bo','go','ro','co','mo','yo']
    plt.figure(figsize=(10,10))
    for c in np.unique(clusters)[1:]:
        plt.plot(Sub_space_2[np.where(clusters==c),0],Sub_space_2[np.where(clusters==c),1],colors[c%6])
    plt.plot(Sub_space_2[np.where(clusters==-1),0],Sub_space_2[np.where(clusters==-1),1], 'ko')
    plt.axis([0,20,np.min(Sub_space_2[:,1]),5])
    plt.show()