import time
import sys
import getopt
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")

from Utilities import *
from MF import *
from Visualization_utilities import *
from transformation import Transformation
from Signatures import Signature


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
        dp = mesh.sample_points_uniformly(number_of_points=19900)
        dp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
        dp=dp.voxel_down_sample(voxel_size=voxel_down_sample)
        #Visualizacion
        if visualization==True:
            o3d.visualization.draw_geometries([dp])
    return dp

def Detect_simetries(path="D:\\Documentos\\INNOVATE\\lib\\symmetry_detection_python\\Lineas 01_10.pts",
                    visualization=True,
                    geometry_type="pointCloud",
                    voxel_down_sample=0.02,
                    NN_for_signature_build=30,
                    random_frac=0.1/16,
                    filtered_SS=False,
                    Cluster_min_samples=30,
                    Cluster_xi=0.001):
    #LOAD GEOMETRY
    dp=Geometry_load(path=path,visualization=visualization,voxel_down_sample=voxel_down_sample,geometry_type=geometry_type)

    star_time=time.time()
    #BUILD SIGNATURES
    indexes=np.arange(np.asarray(dp.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(dp)
    signatures=Signature(dp,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                        rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/4,
                        NN=NN_for_signature_build)

    signature_time=time.time()
    print("numero de puntos")
    print(signatures.shape)
    print("tiempo de construccion de signatures")
    print(signature_time-star_time)
    signature_time=signature_time-star_time
    #Visualización
    if visualization==True:
        print("signatures")
        graficarPropiedad(signatures,dp,'sphericity',frac=1)
    
    #FILTER DISCONTINUITIES
    t1=time.time()
    #signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)
    signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.5,radius=0.02)
    t2=time.time()
    print("tiempo de construccion de signatures")
    print(t2-t1)
    filter_time=t2-t1

    #Visualización
    if visualization==True:
        print("ratio")
        graficarPropiedad(signaturesp,dp,'ratio',frac=1)
    #MENSAJES
    print("Non discontinuitinual points")
    print(len(signaturesp))

    #POINT PRUNNING
    t1=time.time()
    pp=prune_points(signaturesp,"sphericity",0.005)
    t2=time.time()
    print("tiempo de construccion de seleccion")
    print(t2-t1)
    prunning_time=t2-t1

    #MENSAJES
    print("Pruned points")
    print(len(pp))
    #Visualización
    if visualization==True:
        print("Puntos seleccionados")
        graficarPropiedad(pp,dp,'sphericity',frac=1)

    #RANDOM SAMPLING
    t1=time.time()
    ppp=random_sample(pp,random_frac)
    t2=time.time()
    print("tiempo de construccion de seleccion random")
    print(t2-t1)
    random_time=t2-t1

    print("Random points")
    print(len(ppp))
    #Visualizacion
    if visualization==True:
        print("Puntos aleatorios")
        graficarPropiedad(ppp,dp,'sphericity',frac=1)

    #TRANSFORMATION SPACE BUILDING
    t1=time.time()
    [trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=10000.6,rand_samp_percentage=random_frac,
                                             rigid=True,only_reflections=True,NN=80)

    t2=time.time()
    print("tiempo de construccion de seleccion random")
    print(t2-t1)
    transformation_time=t2-t1

    z=0
    #Mensajes
    print("numero de puntos en el espacio de transformadas")
    print(len(trans_s))
    print('desviacion estandar z=0')
    std=np.std(Transformation().v_toPoint(trans_s[:,z]))
    print(std)
    print(np.std(Transformation().v_toPoint(trans_s[:,z]),axis=1))

    #TRANSFORMATION SPACE PRUNNING----------------------------------------------------
    filter2_time=0
    if filtered_SS==True:
        t1=time.time()
        diagonal_length=np.linalg.norm(dp.get_max_bound()-dp.get_min_bound())
        pTrans0=pruneTransPoints(trans_s=trans_s[:,0],
                        Rx_th=0.001*math.pi/(math.pi**2),
                        Ry_th=0.1*math.pi/(math.pi**2),
                        Rz_th_min=0.1*math.pi/(math.pi**2),
                        Rz_th_max=0.9*math.pi/(math.pi**2),
                        Tz_th=0.001/(4/((diagonal_length)**2)),
                        #Ty_th=0.0001/(4/((diagonal_length)**2))
                        )
        pTrans1=pruneTransPoints(trans_s=trans_s[:,1],
                Rx_th=0.001*math.pi/(math.pi**2),
                Ry_th=0.1*math.pi/(math.pi**2),
                Rz_th_min=0.1*math.pi/(math.pi**2),
                Rz_th_max=0.9*math.pi/(math.pi**2),
                Tz_th=0.001/(4/((diagonal_length)**2)),
                #Ty_th=0.0001/(4/((diagonal_length)**2))
                )
        print(pTrans1.shape)
        trans_s=[pTrans0.reshape(-1,1),pTrans1.reshape(-1,1)]
        t2=time.time()
        print("Tiempo de filtrado")
        print(t2-t1)
        filter2_time=t2-t1

        t1=time.time()
        [Clusters,points,Clust_set]=run_clustering_OPTICS(min_samples=30,max_eps=std,xi=0.001,only_reflections=False,Trans_Space=trans_s[z])
        z=0
        t2=time.time()
        print("tiempo de clustering")
        print(t2-t1)
        clustering_time=t2-t1

        print("found Clusters")
        print(Clusters.shape)
        print(Clusters)
        print(Clusters[z,:].max())

        print("Cluster_hierarchy z")
        print(Clust_set[z].cluster_hierarchy_)

        #Visualizacion
        if visualization==True:
            reachability_plot(Clust_set,trans_s[z],z=0)
            colors=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5]]
            n_components=3
            plot_Clusters(z,trans_s[z],Clusters,colors,n_components)

            graf_cluster(ini=0,fin=3,rad=0.2,z=0,cluster=Clusters[z,:],
                        transformation_space=trans_s[z][:,0],point_cloud=dp,KDT=pt)
        cluster_points=[]
        for i in range(3):
            trans_cluster=get_cluster_transformation_points(Clusters[z,:],i,trans_s[z][:,z])
            sig_cluster=get_signatures_from_transformation(signatures,trans_cluster)
            point_index_cluster=get_cluster_NN_points_index_from_signatures(sig_cluster,dp,pt,20)
            if visualization==True:
                cluster_cloud=build_pointcloud_simetrie(dp,point_index_cluster)
                o3d.visualization.draw_geometries([cluster_cloud])
            cluster_points.append(point_index_cluster)

    else:
        print(trans_s.shape)
        t1=time.time()
        [Clusters,points,Clust_set]=run_clustering_OPTICS(min_samples=30,max_eps=std,xi=0.001,Trans_Space=trans_s)
        t2=time.time()
        print("tiempo de clustering")
        print(t2-t1)
        clustering_time=t2-t1

        print("found Clusters")
        print(Clusters.shape)
        print(Clusters[:,z].max())

        print("Cluster_hierarchy z")
        print(Clust_set[z].cluster_hierarchy_)

        #Visualizacion
        if visualization==True:
            reachability_plot(Clust_set,trans_s)
            colors=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[0.5,0.5,0.5]]
            n_components=3
            plot_Clusters(z,trans_s,Clusters,colors,n_components)

            graf_cluster(ini=0,fin=3,rad=0.2,z=0,cluster=Clusters[:,z],
                        transformation_space=trans_s[:,0],point_cloud=dp,KDT=pt)
        cluster_points=[]
        for i in range(3):
            trans_cluster=get_cluster_transformation_points(Clusters[:,z],i,trans_s[:,z])
            sig_cluster=get_signatures_from_transformation(signatures,trans_cluster)
            point_index_cluster=get_cluster_NN_points_index_from_signatures(sig_cluster,dp,pt,20)
            if visualization==True:
                cluster_cloud=build_pointcloud_simetrie(dp,point_index_cluster)
                o3d.visualization.draw_geometries([cluster_cloud])
            cluster_points.append(point_index_cluster)

    print("tiempo total")
    print(signature_time+filter_time+prunning_time+random_time+transformation_time+filter2_time+clustering_time)
    return cluster_points

def Detect_Tube_NonLinear_PCA(path="D:\\Documentos\\INNOVATE\\lib\\symmetry_detection_python\\Lineas 01_10.pts",
                                visualization=True,
                                geometry_type="pointCloud",
                                voxel_down_sample=0.02,
                                NN_for_signature_build=30,
                                random_frac=0.1/16,
                                filtered_SS=False,
                                bandwidth=0.05,
                                min_bin_freq=80):
    #LOAD GEOMETRY
    dp=Geometry_load(path=path,visualization=visualization,geometry_type=geometry_type)
    
    t1=time.time()
    #GENERACION DE SIGNATURES
    print(np.asarray(dp.points).shape)
    indexes=np.arange(np.asarray(dp.points).shape[0])
    pt = o3d.geometry.KDTreeFlann(dp)
    signatures=Signature(dp,pt,indexes)
    Signature().v_build(signatures,NN_Criteria="KNN",
                        rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/8,
                        NN=NN_for_signature_build)

    t2=time.time()
    print("tiempo de construccion de signatures")
    print(t2-t1)
    signature_time=t2-t1

    #Visualización
    if visualization==True:
        print("signatures")
        graficarPropiedad(signatures,dp,'sphericity',frac=1)

    #FILTRO DE PUNTOS DISCONTINUOS
    t1=time.time()
    signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)
    t2=time.time()
    print("tiempo de construccion de filtrado de discontinuidades")
    print(t2-t1)
    filter_time=t2-t1

    #Visualización
    if visualization==True:
        print("ratio")
        graficarPropiedad(signaturesp,dp,'ratio',frac=1)

    print("Total points")
    print(len(signatures))
    print("Non discontinuitinual points")
    print(len(signaturesp))

    #FILTRO DE PUNTOS UMBILICALES
    t1=time.time()
    pp=prune_points(signaturesp,"sphericity",0.0005) #Vectorizar
    t2=time.time()
    print("tiempo de construccion de seleccion")
    print(t2-t1)
    prunning_time=t2-t1

    print("Pruned points")
    print(len(pp))

    #Visualización
    if visualization==True:
        print("ratio")
        graficarPropiedad(pp,dp,'sphericity',frac=1)

    #MUESTREO ALEATORIO
    t1=time.time()
    ppp=random_sample(pp,random_frac)
    t2=time.time()
    print("tiempo de construccion de seleccion random")
    print(t2-t1)
    random_time=t2-t1

    #Visualización
    if visualization==True:
        print("ratio")
        graficarPropiedad(ppp,dp,'sphericity',frac=1)

    print("numero de puntos aleatorios")
    print(len(ppp))

    #GENERACION DE ESPACION DE TRANSFORMACIONES
    t1=time.time()
    [trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=100000.6,rand_samp_percentage=random_frac,
                                                rigid=True,only_reflections=True,NN=80)
    t2=time.time()
    print("tiempo de construccion de generacion de espacio de transformadas")
    print(t2-t1)
    transformation_time=t2-t1

    z=0
    print("numero de puntos en el espacio de transformadas")
    print(len(trans_s))
    print('desviacion estandar z=0')
    std=np.std(Transformation().v_toPoint(trans_s[:,z]))
    print(std)
    print(np.std(Transformation().v_toPoint(trans_s[:,z]),axis=0))

    filter2_time=0
    if filtered_SS==True:
        t1=time.time()
        diagonal_length=np.linalg.norm(dp.get_max_bound()-dp.get_min_bound())
        pTrans=pruneTransPoints(trans_s=trans_s[:,z],
                        Rx_th=0.001*math.pi/(math.pi**2),
                        Ry_th=0.001*math.pi/(math.pi**2),
                        Rz_th_min=0.1*math.pi/(math.pi**2),
                        Rz_th_max=0.9*math.pi/(math.pi**2),
                        Tz_th=0.001/(4/((diagonal_length)**2)),
                        #Ty_th=0.0001/(4/((diagonal_length)**2))
                        )
        t2=time.time()
        print("Tiempo de filtrado")
        print(t2-t1)
        filter2_time=t2-t1
    else:
        pTrans=trans_s[:,z]

    if visualization==True:
        display_trans_prunning(trans_s[:,z],pTrans)

    #GENERACION DE SUB-ESPACIOS
    Sub_space_1=get_SSpace1(Transformation().v_toPoint(pTrans),dp)
    Sub_space_2=get_SSpace2(Transformation().v_toPoint(pTrans),dp)

    #GRAFICAS DE PROYECCIONES NO LINEALES
    if visualization==True:
        display_sub_space(Sub_space_1)

    #COMPUTE CLUSTERS
    t1=time.time()
    Cluster=MeanShift(bandwidth=bandwidth,n_jobs=4,cluster_all=False,bin_seeding=True,min_bin_freq=min_bin_freq)
    t2=time.time()
    print("tiempo de clustering")
    print(t2-t1)
    clustering_time=t2-t1

    clusters=Cluster.fit_predict(Sub_space_2)
    print("Number of clusters")
    print(np.max(clusters))

    #GRAFICA DE CLUSTERS EN ESPACIO PROYECTADO
    if visualization==True:
        display_cluster(clusters,Sub_space_2)

    #GRAFICA DE CLUSTERS EN LA NUBE DE PUNTOS
    if visualization==True:
        graf_cluster(ini=0,fin=3,rad=0.1,z=z,cluster=clusters,
                            transformation_space=pTrans,point_cloud=dp,KDT=pt)

    cluster_points=[]
    for i in range(3):
        trans_cluster=get_cluster_transformation_points(clusters,i,trans_s[:,z])
        sig_cluster=get_signatures_from_transformation(signatures,trans_cluster)
        point_index_cluster=get_cluster_NN_points_index_from_signatures(sig_cluster,dp,pt,20)
        if visualization==True:
            cluster_cloud=build_pointcloud_simetrie(dp,point_index_cluster)
            o3d.visualization.draw_geometries([cluster_cloud])
        cluster_points.append(point_index_cluster)

    print("tiempo total")
    print(signature_time+filter_time+prunning_time+random_time+transformation_time+filter2_time+clustering_time)
    return cluster_points
#TERMINAR RETURN
    #return []