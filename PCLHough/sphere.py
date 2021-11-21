from Utils import *

class Sphere():
#    def __init__(self):
#      return 0
  # direction vectors
#        self.vertices=np.empty()
  # surface triangles
        #self.triangles
  
  # creates nodes and edges of icosahedron
    def getDirections(self,subDivisions):
      sub=subDivisions
      dx=np.arange(-1,1+sub,sub)
      x,y,z=np.meshgrid(dx,dx,dx)
      p=np.hstack((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)))
      self.vertices=np.delete(p,np.where(np.prod(np.equal(p,np.array([0,0,0])),axis=1)==1)[0],axis=0)
      self.vertices=np.vectorize(lambda p:p/np.linalg.norm(p),signature="(j)->(i)")(self.vertices)


  # make vectors nondirectional and unique
    def makeUnique(self):
      # make hemisphere
      #del_index=np.vectorize(pyfunc=(lambda v,i: i if v[2]<0 else -1),signature="(a,b),(j)->()")(self.vertices,np.arange(self.vertices.shape[0]))
      #del_index=del_index[del_index>=0]
      hemisphere_vertices=np.delete(self.vertices,np.where((self.vertices[:,2]>0)==True)[0],axis=0)
      # update indices

      # make equator vectors unique
      v=hemisphere_vertices
      td=np.logical_and(v[:,2]==0,np.logical_or(v[:,0]<0,np.logical_and(v[:,0]==0 , v[:,2]==-1)))
      hemisphere_vertices=np.delete(hemisphere_vertices,np.where(td==True)[0],axis=0)
      # update indices

      self.vertices=hemisphere_vertices

class Disc():
#    def __init__(self):
#      return 0
  # direction vectors
#        self.vertices=np.empty()
  # surface triangles
        #self.triangles
  
  # creates nodes and edges of icosahedron
    def getDirections(self,subDivisions):
      sub=subDivisions
      theta=np.arange(0,np.pi+np.pi*sub,np.pi*sub)
      v=np.array([[1],[0]])
      R = np.array(((np.cos(theta), -np.sin(theta)),
              (np.sin(theta), np.cos(theta))))
      v=np.matmul(v.T,R).shape
      v=v.T.reshape(-1,2)
      p=np.hstack((v,np.zeros((v.shape[0],1))))
      p=np.hstack((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)))
      self.vertices=np.delete(p,np.where(np.prod(np.equal(p,np.array([0,0,0])),axis=1)==1)[0],axis=0)
      self.vertices=np.vectorize(lambda p:p/np.linalg.norm(p),signature="(j)->(i)")(self.vertices)


  # make vectors nondirectional and unique
    def makeUnique(self):
      # make hemisphere
      #del_index=np.vectorize(pyfunc=(lambda v,i: i if v[2]<0 else -1),signature="(a,b),(j)->()")(self.vertices,np.arange(self.vertices.shape[0]))
      #del_index=del_index[del_index>=0]
      hemisphere_vertices=np.delete(self.vertices,np.where((self.vertices[:,2]>0)==True)[0],axis=0)
      # update indices

      # make equator vectors unique
      v=hemisphere_vertices
      td=np.logical_and(v[:,2]==0,np.logical_or(v[:,0]<0,np.logical_and(v[:,0]==0 , v[:,2]==-1)))
      hemisphere_vertices=np.delete(hemisphere_vertices,np.where(td==True)[0],axis=0)
      # update indices

      self.vertices=hemisphere_vertices