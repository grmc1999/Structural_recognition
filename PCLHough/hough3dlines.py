from Utils import orthogonal_LSQ
from numpy import subtract
from Utils import *
from hough import *
pointcloud


def hough3D(
    path="",
    opt_dx=0,
    opt_minvotes=0,
    opt_nlines=2,
    granularity = 4,
            ):

    num_directions = [12, 21, 81, 321, 1281, 5121, 20481]
    pcl=o3d.io.read_point_cloud(path)
    X=pointcloud.PointCloud(pcl)
    X.setPCL(pcl)

    X.shiftToOrigin()
    X.getMinMax3D()

    d=X.diagonal_length
    minPshifted=X.minBound
    maxPshifted=X.maxBound

    if opt_dx==0:
        opt_dx=d/64


    hough=Hough(minPshifted,maxPshifted,opt_dx,granularity)

    hough.add(X)

    Y=pointcloud.PointCloud()
    nlines=0
    A=[]
    B=[]

    while X.points.shape[0]>1 and (opt_nlines==0 or opt_nlines>nlines):

        hough.subtract(Y)

        nvotes=hough.getLine()
        a=hough.a
        b=hough.b

        X.pointsCloseToLine(a,b,opt_dx,Y)

        a,b,rc=orthogonal_LSQ(Y)
        if rc==0:
            break

        X.pointsCloseToLine(a,b,opt_dx,Y)
        nvotes=Y.points.shape[0]
        if nvotes < opt_minvotes:
            break

        a,b,rc=orthogonal_LSQ(Y)
        if rc==0:
            break
        
        a=a+X.shift

        nlines=nlines+1
        if rc==0:
            break

        X.removePoints(Y)
    return A,B

def hough3D_pcl(
    pcl,
    opt_dx=0,
    opt_minvotes=0,
    opt_nlines=2,
    granularity = 1/8,
            ):

    num_directions = [12, 21, 81, 321, 1281, 5121, 20481]
    pcl=pcl
    X=pointcloud.PointCloud()
    X.setPCL(pcl)

    X.shiftToOrigin()
    X.getMinMax3D()

    d=X.diagonal_length
    minPshifted=X.minBound
    maxPshifted=X.maxBound

    if opt_dx==0:
        opt_dx=d/(64)
    print("opt_dx")
    print(opt_dx)


    hough=Hough(minPshifted,maxPshifted,opt_dx,granularity)
    #hough=Hough(minPshifted,maxPshifted,granularity)

    hough.add(X)

    ivs=hough.VotingSpace

    Y=pointcloud.PointCloud()
    nlines=0
    A=[]
    B=[]

    while X.points.shape[0]>1 and (opt_nlines==0 or opt_nlines>nlines):
        print(A)
        print(B)

        hough.subtract(Y)

        nvotes=hough.getLine()
        a=hough.a
        b=hough.b

        X.pointsCloseToLine(a,b,hough.num_x,nvotes,Y)

        o3d.visualization.draw_geometries([Y.pcl])

        a,b,rc=orthogonal_LSQ(Y)

        line=draw_line(a,b,300)
        o3d.visualization.draw_geometries([Y.pcl,line])

        if rc==0:
            break

        X.pointsCloseToLine(a,b,hough.num_x,nvotes,Y)
        nvotes=Y.points.shape[0]
        if nvotes < opt_minvotes:
            break

        a,b,rc=orthogonal_LSQ(Y)

        line=draw_line(a,b,300)
        o3d.visualization.draw_geometries([Y.pcl,line])

        if rc==0:
            break
        
        a=a+X.shift
        A.append(a)
        B.append(b)

        nlines=nlines+1
        if rc==0:
            break

        X.removePoints(Y)

        line=draw_line(a,b,300)
        o3d.visualization.draw_geometries([X.pcl,line])

    return A,B,nlines,ivs

