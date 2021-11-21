import sys
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")
from Utilities import *
from MF import *
from Visualization_utilities import *
from transformation import Transformation
from Signatures import Signature
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#CARGA DE GEOMETRIA
mesh = o3d.io.read_triangle_mesh("D:\Documentos\INNOVATE\lib\symmetry-detection-reflection\mesh\Wine_Bottle.obj")
dp = mesh.sample_points_uniformly(number_of_points=20000)
dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(20))
#dp=dp.voxel_down_sample(voxel_size=0.15)
o3d.visualization.draw_geometries([dp])

# p=o3d.io.read_point_cloud('D:\Documentos\INNOVATE\lib\symmetry_detection_python\Linea 22.pts')
# dp=p
# dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([dp],point_show_normal=False)

# def display_inlier_outlier(cloud, ind):
#     inlier_cloud = cloud.select_by_index(ind)
#     outlier_cloud = cloud.select_by_index(ind, invert=True)

#     print("Showing outliers (red) and inliers (gray): ")
#     outlier_cloud.paint_uniform_color([1, 0, 0])
#     inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
# cl, ind = dp.remove_statistical_outlier(nb_neighbors=5,std_ratio=1.5)
# display_inlier_outlier(dp, ind)
# dp=cl
# dp=dp.voxel_down_sample(voxel_size=0.02)
# dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([dp])

#GENERACION DE SIGNATURES
print(np.asarray(dp.points).shape)
indexes=np.arange(np.asarray(dp.points).shape[0])
pt = o3d.geometry.KDTreeFlann(dp)
signatures=Signature(dp,pt,indexes)
Signature().v_build(signatures,NN_Criteria="RNN",
                    rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/8,
                    NN=40)
#GRAFICA DE PROPIEDADES
graficarPropiedad(signatures,dp,'ratio',1)
graficarPropiedad(signatures,dp,'sphericity',1)

#FILTRO DE PUNTOS DISCONTINUOS
signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)
print("Total points")
print(len(signatures))
print("Non discontinuitinual points")
print(len(signaturesp))

#FILTRO DE PUNTOS UMBILICALES
pp=prune_points(signaturesp,"sphericity",0.0005) #Vectorizar
#pp=prune_points(signaturesp,"sphericity",0.0001) #Vectorizar
pp=np.array(pp)
graficarPropiedad(pp,dp,'ratio',1)
graficarPropiedad(pp,dp,'sphericity',1)
print("Pruned points")
print(len(pp))

#MUESTREO ALEATORIO
rat=0.1/2
ppp=random_sample(pp,rat)
ppp=np.array(ppp)
graficarPropiedad(ppp,dp,'sphericity',1)
print("numero de puntos aleatorios")
print(len(ppp))

#GENERACION DE ESPACION DE TRANSFORMACIONES
[trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=100000.6,rand_samp_percentage=rat,
                                             rigid=True,only_reflections=True,NN=80)

z=0
print("numero de puntos en el espacio de transformadas")
print(len(trans_s))
print('desviacion estandar z=0')
std=np.std(Transformation().v_toPoint(trans_s[:,0]))
print(std)
print(np.std(Transformation().v_toPoint(trans_s[:,0]),axis=0))
print('desviacion estandar z=1')
print(np.std(Transformation().v_toPoint(trans_s[:,1])))
print(np.std(Transformation().v_toPoint(trans_s[:,1]),axis=0))

diagonal_length=np.linalg.norm(dp.get_max_bound()-dp.get_min_bound())
#FILTRO DE TRANSFORMACIONES
# pTrans=pruneTransPoints(trans_s=trans_s[:,z],
#                     Rx_th=0.01*math.pi/(math.pi**2),
#                     Ry_th=0,
#                     Rz_th_min=0.1*math.pi/(math.pi**2),
#                     Rz_th_max=0.9*math.pi/(math.pi**2),
#                     Tz_th=0.0001,
#                     Tx_th=0.0001
#                     )

pTrans=pruneTransPoints(trans_s=trans_s[:,z],
                    Rx_th=0,
                    Ry_th=0.0001*math.pi/(math.pi**2),
                    Rz_th_min=0.1*math.pi/(math.pi**2),
                    Rz_th_max=0.9*math.pi/(math.pi**2),
                    Tz_th=0.0001/(4/((diagonal_length)**2)),
                    Ty_th=0.0001/(4/((diagonal_length)**2))
                    )

print("numero de puntos en el espacio de transformacion filtrado")
print(pTrans.shape)

#GRAFICA DE HISTOGRAMAS DE PUNTOS
display_Trans_prunning(trans_s[:,z],pTrans)

#ACTUALIZACION DE SIGNATURES
new_pc_ind=np.vectorize(pyfunc=(lambda x: np.array([x.origin_index,x.image_index])),
                                signature='()->(n)')(pTrans)
new_pc_ind=np.unique(new_pc_ind.reshape(1,-1))
signatures=signatures.tolist()
signatures.sort(key=lambda x: getattr(x,"point_index"), reverse=False)
signatures=np.array(signatures)
pruned_signatures=signatures[new_pc_ind]

#GRAFICA DE PUNTOS FILTRADOS
graficarPropiedad(pruned_signatures,dp,property="sphericity",frac=1)

#GENERACION DE SUB-ESPACIOS
Sub_space_1=get_SSpace1(Transformation().v_toPoint(trans_s[:,z]),dp)
Sub_space_2=get_SSpace2(Transformation().v_toPoint(trans_s[:,z]),dp)

#GRAFICAS DE PROYECCIONES NO LINEALES
display_sub_space(Sub_space_1)

#COMPUTE CLUSTERS
Cluster=MeanShift(bandwidth=0.05,n_jobs=4,cluster_all=False,bin_seeding=True,min_bin_freq=80)
clusters=Cluster.fit_predict(Sub_space_2)
print("Number of clusters")
print(np.max(clusters))

#GRAFICA DE CLUSTERS EN ESPACIO PROYECTADO
display_cluster(clusters,Sub_space_2)

#GRAFICA DE CLUSTERS EN LA NUBE DE PUNTOS
graf_cluster(ini=0,fin=3,rad=0.1,z=z,cluster=clusters,
                    transformation_space=trans_s[:,z],point_cloud=dp,KDT=pt)

#GENERACION DE SUB-ESPACIOS FILTRADOS
Sub_space_1=get_SSpace1(Transformation().v_toPoint(pTrans),dp)
Sub_space_2=get_SSpace2(Transformation().v_toPoint(pTrans),dp)

#GRAFICAS DE PROYECCIONES NO LINEALES
display_sub_space(Sub_space_1)

#COMPUTE CLUSTERS
Cluster=MeanShift(bandwidth=0.05,n_jobs=4,cluster_all=False,bin_seeding=True,min_bin_freq=20)
clusters=Cluster.fit_predict(Sub_space_2)
print("Number of clusters")
print(np.max(clusters))
np.where(clusters==-1)

#GRAFICA DE CLUSTERS EN ESPACIO PROYECTADO
display_cluster(clusters,Sub_space_2)

#GRAFICA DE CLUSTERS EN LA NUBE DE PUNTOS
graf_cluster(ini=0,fin=3,rad=0.5,z=z,cluster=clusters,
                    transformation_space=pTrans,point_cloud=dp,KDT=pt)