import sys
import pathlib
pth=str(pathlib.Path().absolute())
print(pth)
sys.path.append(('\\').join(pth.split('\\')[:-1])+"\\Utils")
from Utilities import *
from MF import *
from Visualization_utilities import *
from transformation import Transformation
from Signatures import Signature
import os
import open3d as o3d
import numpy as np


def getRotationMatrix(vect, ang):
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

def getRotationMatrixFromVector(vect1, vect2):
    vect_normal = np.cross(vect1, vect2)
    vect_normal = vect_normal/np.linalg.norm(vect_normal)
    dotProduct = np.dot(vect1/np.linalg.norm(vect1), vect2/np.linalg.norm(vect2))
    ang = np.arccos(dotProduct)
    return getRotationMatrix(vect1, ang)

#pts = 'Beam-W_W8x2_pulg_ac_210.pts'
#pts = 'Beam-W_W8x1000_pulg_mb_368.pts'
pts = 'Beam-W_W4x1301_pulg_mb_261.pts'
#pts = "Beam-W_W8x2102_pulg_ac_208.pts"
dpf=o3d.io.read_point_cloud("E:\\INNOVATE\\DB_TRAINING_1024_PTS\\"+pts.split('_')[0]+"\\"+pts)


ev1,ev2=project_pcl_pca(dpf,
             voxel_size=0.02,
             nb=5,
             std_ratio=1.5,
             p_th=0.1,
             ig="Linearity",
             upper=True
            ) 

rot = getRotationMatrixFromVector(ev1, np.array([0.0, 0.0, 1.0]))
cloud = dpf
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80.0, origin=cloud.get_center())
cloud.rotate(rot)

p = np.asarray(cloud.points)
n_p = np.copy(p)


low_bound = np.min(n_p[:, 2])
high_bound = np.max(n_p[:, 2])
l = high_bound - low_bound
N_SLICES = 10
nd = {i:0 for i in range(N_SLICES)}
pd = {i:[] for i in range(N_SLICES)}

for cont, i in enumerate(n_p):
    for j in range(N_SLICES):
        if i[2]>= low_bound+j*l/N_SLICES and i[2] < low_bound+(j+1)*l/N_SLICES:
            nd[j] += 1
            pd[j].append([n_p[cont, 0], n_p[cont, 1]])
            n_p[cont, 2] = low_bound+j*l/N_SLICES
print(nd)
test = o3d.geometry.PointCloud()
test.points = o3d.utility.Vector3dVector(n_p)
#o3d.visualization.draw_geometries([test])

ww = []
hh = []
mm = []
for i in pd.keys():
    pd[i] = np.array(pd[i])
    co = pd[i].T.dot(pd[i])
    _, v = np.linalg.eig(co)
    aux = pd[i]
    init = np.random.random((aux.shape[0], 3))
    init[:, :2] = aux
    naux = o3d.geometry.PointCloud()
    naux.points = o3d.utility.Vector3dVector(init)
    m_ = np.array([0., 0., np.arctan(v[0, 1]/v[0, 0])])
    mm.append([0., 0., np.arctan(v[0, 1]/v[0, 0])])
    naux.rotate(naux.get_rotation_matrix_from_axis_angle(m_))
    puntex = np.asarray(naux.points)
    w = np.max(puntex[:, 1]) - np.min(puntex[:, 1])
    h = np.max(puntex[:, 0]) - np.min(puntex[:, 0])
    ww.append(w)
    hh.append(h)

    #naux.translate(-naux.get_center())
    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=8.0, origin=np.array([0., 0., 0.]))
    #o3d.visualization.draw_geometries([naux, mesh_frame])
w_out = np.mean(np.array(ww))
h_out = np.mean(np.array(hh))

mm =np.array(mm)
N = len(dpf.points)
if N>= 5000:
    dpf, _ = dpf.remove_statistical_outlier(nb_neighbors=int(N/200), std_ratio = 0.9)
bb = dpf.get_oriented_bounding_box()
R = bb.R

x, y, z = R[0, 0], R[1, 0], R[2, 0] #Extrusion around x axis of bounding box
N = np.array([x, y, z])
P = np.array(dpf.get_center())
s = []
for i in dpf.points:
    v = i-P
    dist = np.dot(v, N)
    #pp = i-dist*N
    s.append(dist)

l = max(s) - min(s)
base = P+N*max(s)
mm_ = np.mean(mm, 0)
print(0.9*w_out, 0.9*h_out, base, l, N, naux.get_rotation_matrix_from_axis_angle(mm_))

