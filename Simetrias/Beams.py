import sys
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")
from Utilities import *
from MF import *
from Visualization_utilities import *
from transformation import Transformation
from Signatures import Signature

dp=Geometry_load(path="D:\\Documentos\\INNOVATE\\Base_de_datos\\beam_test\\_VIGAS0100_pulg_002.pts",
                    visualization=True,
                    voxel_down_sample=0.02,
                    geometry_type="pointCloud")

indexes=np.arange(np.asarray(dp.points).shape[0])
pt = o3d.geometry.KDTreeFlann(dp)
signatures=Signature(dp,pt,indexes)
Signature().v_build(signatures,NN_Criteria="KNN",
                    rad=np.std(np.asarray(dp.points)-np.mean(np.asarray(dp.points),axis=0))/8,
                   NN=30)

print(np.asarray(dp.points).shape)
signaturesp=remove_discontinuities_by_curvature(dp,signatures,pt,curvature=0.4,radius=0.02)
print("Total points")
print(len(signatures))
print("Non discontinuitinual points")
print(len(signaturesp))
pp=prune_points(signaturesp,"sphericity",0.0005)
pp=np.array(pp)
print("Pruned points")
print(len(pp))
rat=0.01
ppp=random.sample(signatures.tolist(),200)
#ppp=random_sample(pp,rat)
ppp=np.array(ppp)
print("numero de puntos aleatorios")
print(len(ppp))

[trans_s,rs,KDT,query]=build_pairing_kd_tree(dp,pp,rad=100000.6,rand_samp_percentage=rat,
                                             rigid=True,only_reflections=True,NN=80)

trans_space=trans_s[:,0]
np_points=Transformation().v_toPoint(trans_space)

import matplotlib.pyplot as pp
val = 0.
pp.figure(figsize=(10,10))
#pp.plot(np.abs(np_points[:,6]), np.zeros_like(np_points[:,6]) + val, 'x')
pp.plot(np.abs(np_points[:,6]), np.abs(np_points[:,4]), 'x')
pp.show()

val = 0.
pp.figure(figsize=(10,10))
#pp.plot(np.abs(np_points[:,6]), np.zeros_like(np_points[:,6]) + val, 'x')
pp.plot(np.abs(np_points[:,6]), np.abs(np_points[:,5]), 'x')
pp.show()

points=np.concatenate(((np.abs(np_points[:,6])).reshape(-1,1), np.abs(np_points[:,4]).reshape(-1,1)),axis=1)
bd=7.5*(((np.max(points[:,0])*1.2-np.min(points[:,0])*1.8)/200+(np.max(points[:,0])*1.2-np.min(points[:,0])*1.8)/200)/2)
modelo_kde=KernelDensity(kernel='epanechnikov',bandwidth=bd).fit(points)
densityPlot(modelo_kde,points)


import seaborn as sns
import pandas as pd
df = pd.DataFrame(np_points, columns = ['k','Rx','Ry','Rz','Tx','Ty','Tz'])
#sns.pairplot(df[['Rx','Ry','Rz','Tx','Ty','Tz']])
sns.pairplot(df[['Rx','Rz']])

pTrans=pruneTransPoints(trans_s=trans_s[:,0],
                #Rx_th=0.001*math.pi/(math.pi**2),
                #Ry_th=0.001*math.pi/(math.pi**2),
                Rz_th_min=0.05*math.pi/(math.pi**2),
                Rz_th_max=0.95*math.pi/(math.pi**2),
                #Tz_th=0.001/(4/((diagonal_length)**2)),
                #Ty_th=0.0001/(4/((diagonal_length)**2))
                )
print(pTrans.shape)
np_points=Transformation().v_toPoint(pTrans)

val = 0.
pp.figure(figsize=(10,10))
#pp.plot(np.abs(np_points[:,6]), np.zeros_like(np_points[:,6]) + val, 'x')
pp.plot(np.abs(np_points[:,6]), np.abs(np_points[:,4]), 'x')
pp.show()

val = 0.
pp.figure(figsize=(10,10))
#pp.plot(np.abs(np_points[:,6]), np.zeros_like(np_points[:,6]) + val, 'x')
pp.plot(np.abs(np_points[:,6]), np.abs(np_points[:,5]), 'x')
pp.show()

val = 0.
pp.figure(figsize=(10,10))
#pp.plot(np.abs(np_points[:,6]), np.zeros_like(np_points[:,6]) + val, 'x')
pp.plot(np.abs(np_points[:,6]), np.abs(np_points[:,1])*np.abs(np_points[:,2]), 'x')
pp.show()

import seaborn as sns
import pandas as pd
df = pd.DataFrame(np_points, columns = ['k','Rx','Ry','Rz','Tx','Ty','Tz'])
#sns.pairplot(df[['Rx','Ry','Rz','Tx','Ty','Tz']])
sns.pairplot(df[['Rx','Rz']])

points=np.concatenate(((np.abs(np_points[:,6])).reshape(-1,1), np.abs(np_points[:,4]).reshape(-1,1)),axis=1)
bd=7.5*(((np.max(points[:,0])*1.2-np.min(points[:,0])*1.8)/200+(np.max(points[:,0])*1.2-np.min(points[:,0])*1.8)/200)/2)
modelo_kde=KernelDensity(kernel='epanechnikov',bandwidth=bd).fit(points)
densityPlot(modelo_kde,points)