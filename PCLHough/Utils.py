import sys
import pathlib
import numpy as np
import open3d as o3d


def orthogonal_LSQ(pointCloud):
    points=np.asarray(pointCloud.points)
    a=np.mean(points,axis=0)
    centered=points-a
    scatter=np.matmul(centered.T,centered)
    v,V=np.linalg.eig(scatter)
    b=V[:,np.argmax(v)]
    rc=np.max(v)
    return a,b,rc

def draw_line(a,b,L=100):
    p1=a+b*L
    p2=a+b*(-L)
    #print(p1)
    #print(p2)
    points=[p1.tolist(),p2.tolist()]
    #points=list(np.vectorize(lambda x:list(x),otypes=[object],signature="(j)->()")(pp))
    #print(points)
    line=[[0,1]]
    colors = [[1, 0, 0] for i in range(len(line))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([line_set])
    return line_set