from Utils import *

pth=str(pathlib.Path().absolute())

#POINT CLOUD FUNCIONS

class PointCloud():
    def __init__(self):
  # points of the point cloud
        self.pcl = o3d.geometry.PointCloud()
        self.pcl.points = o3d.utility.Vector3dVector(np.array([[0,0,0]]))
        self.points=np.asarray(self.pcl.points)
  # translation of pointCloud as done by shiftToOrigin()
    def setPCL(self,pcl):
        self.pcl=pcl
        self.points=np.asarray(self.pcl.points)
  # translate point cloud so that center = origin
    def shiftToOrigin(self):
        point_arr=np.asarray(self.pcl.points)
        self.pcl = o3d.geometry.PointCloud()
        self.pcl.points = o3d.utility.Vector3dVector(point_arr-np.sum(point_arr,axis=0)/point_arr.shape[0])
        self.points=point_arr-np.sum(point_arr,axis=0)/point_arr.shape[0]
        self.shift=np.sum(point_arr,axis=0)/point_arr.shape[0]
  # mean value of all points (center of gravity)
    def meanValue(self):
        return np.sum(np.asarray(self.points),axis=0)/np.asarray(self.points).shape[0]
  # bounding box corners
    def getMinMax3D(self):
        self.minBound=self.pcl.get_min_bound()
        self.maxBound=self.pcl.get_max_bound()
        self.diagonal_length=np.linalg.norm(self.pcl.get_max_bound()-self.pcl.get_min_bound())
  # store points closer than dx to line (a, b) in Y
    def pointsCloseToLine(self,a,b,num_x,nvotes,Y):
        points_arr=np.asarray(self.pcl.points)
        t=np.dot(b,(points_arr-a).T)

        d=points_arr-(a.reshape(-1,3)+(t.reshape(-1,1)*b.reshape(-1,3)))
        #close_points_arr=points_arr[np.where(np.linalg.norm(d,axis=1)<=dx),:]
        close_points_arr=np.array(sorted(list(np.hstack((points_arr,np.linalg.norm(d,axis=1).reshape(-1,1)))), key=lambda x: x[-1], reverse=False))
        close_points_arr=np.delete(close_points_arr,np.arange(0,nvotes+1).astype(int),axis=0)


        new_PCL=o3d.geometry.PointCloud()
        new_PCL.points=o3d.utility.Vector3dVector(close_points_arr[:,:3])
        Y.setPCL(new_PCL)       
  # removes the points in Y from PointCloud
  # WARNING: only works when points in same order as in pointCloud!
    def removePoints(self,Y):
        point_set=self.points #3,m
        point_subset=Y.points #3,n ;n<m
        index_subset=np.vectorize((lambda point_set,i: np.where(np.prod(point_set==i,axis=1)==1)[0]),
                    signature="(x,y),(j)->()")(point_set,point_subset)
        point_set=np.delete(point_set,index_subset,axis=0)

        new_PCL=o3d.geometry.PointCloud()
        new_PCL.points=o3d.utility.Vector3dVector(point_set)

        self.setPCL(new_PCL)

