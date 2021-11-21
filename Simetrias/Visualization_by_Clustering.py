import sys
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")

from Utilities import *
from Visualization_utilities import *
import random
from MF import *
from transformation import Transformation
from Signatures import Signature

path="D:\Documentos\INNOVATE\lib\symmetry_detection_python\Linea 12.pts"
dp=Geometry_load(path=path,
                visualization=True,
                voxel_down_sample=0.01,
                geometry_type="pointCloud")

indexes=np.arange(np.asarray(dp.points).shape[0])
pt = o3d.geometry.KDTreeFlann(dp)
signatures=Signature(dp,pt,indexes)
Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/4,
                    NN=30)

signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)

print(len(signaturesp))
#pp=prune_points(signaturesp,"sphericity",0.0005)
pp=prune_points(signaturesp,"sphericity",0.0001)
print("numero de puntos umbilicales")
print(len(pp))

rat=0.1/32
ppp=random_sample(pp,rat)
print("numero de puntos aleatorios")
print(len(ppp))
np.asarray(dp.points).shape

[trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=10000.6,rand_samp_percentage=rat,
                                             rigid=True,only_reflections=True,NN=60)

z=0
print("numero de puntos en el espacio de transformadas")
print(len(trans_s))
print('desviacion estandar z='+str(z))
std=np.std(Transformation().v_toPoint(trans_s[:,z]))
print(std)
print(np.std(Transformation().v_toPoint(trans_s[:,z]),axis=1))

ClusterMethod="OPTICS"
if ClusterMethod=="Meanshift":
    [Clusters,points]=run_clustering_Meanshift(trans_s,BW=std/2,min_bin_freq=1)
elif ClusterMethod=="DBSCAN":
    [Clusters,points]=run_clustering_DBSCAN(Trans_Space=trans_s,min_samples_=60,eps_=0.07)
elif ClusterMethod=="OPTICS":
#    [Clusters,points,Clust_set]=run_clustering_OPTICS(min_samples=60,max_eps=std,xi=0.0,Trans_Space=trans_s)
    #[Clusters,points,Clust_set]=run_clustering_OPTICS(min_samples=30,max_eps=std,xi=0.0,Trans_Space=trans_s)
    [Clusters,points,Clust_set]=run_clustering_OPTICS(min_samples=30,max_eps=std,xi=0.001,Trans_Space=trans_s)


    print("Cluster_hierarchy z="+str(z))
    print(Clust_set[z].cluster_hierarchy_)

print(Clusters.shape)
print(Clusters[:,0].max())
print(Clusters[:,1].max())

reachability_plot(Clust_set,trans_s)

for i in range(2):
    trans_cluster=get_cluster_transformation_points(Clusters[:,z],i,trans_s[:,z])
    sig_cluster=get_signatures_from_transformation(signatures,trans_cluster)
    cluster_NN_points=get_cluster_NN_points_index_from_signatures(sig_cluster,dp,pt,15)
    cluster_cloud=build_pointcloud_simetrie(dp,cluster_NN_points)
    o3d.visualization.draw_geometries([cluster_cloud])

colors=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5]]
n_components=3

plot_Clusters(0,trans_s,Clusters,colors,n_components)
plot_Clusters(1,trans_s,Clusters,colors,n_components)