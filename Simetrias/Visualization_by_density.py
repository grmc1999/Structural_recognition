import sys
import getopt
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")
from Visualization_utilities import *
from Utilities import *
from MF import *
from transformation import Transformation
from Signatures import Signature

#GEOMETRY LOAD
path="D:\Documentos\INNOVATE\lib\symmetry_detection_python\Linea 12.pts"
dp=Geometry_load(path=path,
                visualization=True,
                voxel_down_sample=0.01,
                geometry_type="pointCloud")
#SIGNATURE GENERATION
indexes=np.arange(np.asarray(dp.points).shape[0])
pt = o3d.geometry.KDTreeFlann(dp)
signatures=Signature(dp,pt,indexes)
Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/4,
                    NN=30)
#REMOVE DISCONTINUITIES
signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)
print(len(signaturesp))
#POINT PRUNNING
pp=prune_points(signaturesp,"sphericity",0.0001)
print("numero de puntos umbilicales")
print(len(pp))
#RANDOM SAMPLE
rat=0.1/32
ppp=random_sample(pp,rat)
print("numero de puntos aleatorios")
print(len(ppp))
np.asarray(dp.points).shape
#TRANSFORMATION SPACE GENERATION
[trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=10000.6,rand_samp_percentage=rat,
                                             rigid=True,only_reflections=True,NN=50)
print("numero de puntos en el espacio de transformadas")
print(len(trans_s))
z=0
points=Transformation().v_toPoint(trans_s[:,z])
print("z=0")
print(points)

embedding= MDS(n_components=2,n_jobs=4)
points = embedding.fit_transform(points)
print("z=0")
print(points)
plot=np.concatenate((np.zeros((points.shape[0],1)),points0),axis=1)

bd=4.5*(((np.max(points0[:,0])*1.2-np.min(points0[:,0])*1.8)/200+(np.max(points0[:,0])*1.2-np.min(points0[:,0])*1.8)/200)/2)
print(bd)

modelo_kde=KernelDensity(kernel='epanechnikov',bandwidth=bd).fit(points)
densityPlot(modelo_kde,points)
