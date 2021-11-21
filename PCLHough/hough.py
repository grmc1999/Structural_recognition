from numpy import pi
import Utils
from sphere import *
import pointcloud

class Hough():
      def __init__(self,minP,maxP,var_dx=0,sphereGranularity=1/8):
            self.sphere=Sphere()
            self.sphere.getDirections(sphereGranularity)
            self.num_b=self.sphere.vertices.shape[0]

        # Compute x'y' discretization
            self.max_x=max(np.linalg.norm(minP),np.linalg.norm(maxP))
            range_x=self.max_x*2
            self.dx=var_dx
            if var_dx==0:
                  self.dx=range_x/64
            #REVISAR
            self.num_x=round(range_x/self.dx)
            self.VotingSpace=np.zeros((1,self.num_x*self.num_x*self.num_b)) #3,num_x * num_x * num_b  3,Y' x X' x B
            self.VotingSpace=self.VotingSpace.reshape(self.num_b,self.num_x,self.num_x)

      #def delHough(self):

  # add all points from point cloud to voting space
      def perPointAdd(self,point):
            self.pointVote(point,True)
      
      def add(self,pointCloud):
            np.vectorize(self.perPointAdd,signature="(j)->()")(pointCloud.points)
  # subtract all points from point cloud to voting space
      def perPointSubtract(self,point):
            self.pointVote(point,False)

      def subtract(self,pointCloud):
            np.vectorize(self.perPointSubtract,signature="(j)->()")(pointCloud.points)

  # add or subtract (add==false) one point from voting space
      def perVertexPointVote(self,point,vertex,add):
            b=vertex
            if b[2]!=-1:
                  beta = 1 / (1 + b[2])
            else:
                  beta=1e-50
            x_new = ((1 - (beta * (b[0] * b[0]))) * point[0]) - ((beta * (b[0] * b[1])) * point[1]) - (b[0] * point[2])
            y_new = ((-beta * (b[0] * b[1])) * point[0]) + ((1 - (beta * (b[1] * b[1]))) * point[1]) - (b[1] * point[2])
            x_i = round((x_new + self.max_x) / self.dx)
            y_i = round((y_new + self.max_x) / self.dx)
            return np.array([b[0],b[1],b[2],x_i,y_i,(int(add)*2-1)])

      def updateVotingSpace(self,bxyv):
            bi=np.where(np.prod(np.equal(self.sphere.vertices,bxyv[:3]),axis=1))[0]
            self.VotingSpace[bi,int(bxyv[3]),int(bxyv[4])]=self.VotingSpace[bi,int(bxyv[3]),int(bxyv[4])]+bxyv[5]

      def pointVote(self,point,add):
            sub_pre_voting_space=np.vectorize(self.perVertexPointVote,signature="(k),(i),()->(j)")(point,self.sphere.vertices,add) # b x y vote ,n_vertices
            uniq=np.unique(sub_pre_voting_space,axis=0)
            sub_pre_voting_space=np.vectorize(lambda A,uniq_i: np.hstack((uniq_i,np.sum(A[np.prod(np.equal(A[:,:3],uniq_i[:3]),axis=1).astype(bool)][:,4]))),
                                          signature="(a,b),(i)->(j)")(sub_pre_voting_space,uniq)

            np.vectorize(self.updateVotingSpace,signature="(j)->()")(sub_pre_voting_space)


  # returns the line with most votes (rc = number of votes)
      def getLine(self):
            votes=np.max(self.VotingSpace)
            b_i,x_i,y_i=np.where(self.VotingSpace==np.max(self.VotingSpace))
            print(b_i)
            print(x_i)
            print(y_i)

            print("found "+str(b_i.shape[0])+" directions")
            if b_i.shape[0]>1:
                  #print("found "+str(b_i.shape[0])+" directions")
                  b_i=b_i[0]
                  x_i=x_i[0]
                  y_i=y_i[0]
            self.a=np.zeros((1,3))
            self.b=self.sphere.vertices[b_i,:].reshape(-1,)
            print(self.b)
            self.a[0,0] = x_i * (1 - ((self.b[0] * self.b[0]) / (1 + self.b[2]))) - y_i * ((self.b[0] * self.b[1]) / (1 + self.b[2]))
            self.a[0,1] = x_i * (-((self.b[0] * self.b[1]) / (1 + self.b[2]))) + y_i * (1 - ((self.b[1] * self.b[1]) / (1 + self.b[2])))
            self.a[0,2] = - x_i * self.b[0] - y_i * self.b[1]
            return votes

class Planar_Hough():
      def __init__(self,minP,maxP,var_dx=0,sphereGranularity=1/8):
            self.sphere=Sphere()
            self.sphere.getDirections(sphereGranularity)
            self.num_b=self.sphere.vertices.shape[0]

        # Compute x'y' discretization
            self.max_x=max(np.linalg.norm(minP),np.linalg.norm(maxP))
            range_x=self.max_x*2
            self.dx=var_dx
            if var_dx==0:
                  self.dx=range_x/64
            #REVISAR
            self.num_x=round(range_x/self.dx)
            self.VotingSpace=np.zeros((1,self.num_x*self.num_x*self.num_b)) #3,num_x * num_x * num_b  3,Y' x X' x B
            self.VotingSpace=self.VotingSpace.reshape(self.num_b,self.num_x,self.num_x)

      #def delHough(self):

  # add all points from point cloud to voting space
      def perPointAdd(self,point):
            self.pointVote(point,True)
      
      def add(self,pointCloud):
            np.vectorize(self.perPointAdd,signature="(j)->()")(pointCloud.points)
  # subtract all points from point cloud to voting space
      def perPointSubtract(self,point):
            self.pointVote(point,False)

      def subtract(self,pointCloud):
            np.vectorize(self.perPointSubtract,signature="(j)->()")(pointCloud.points)

  # add or subtract (add==false) one point from voting space
      def perVertexPointVote(self,point,vertex,add):
            b=vertex
            if b[2]!=-1:
                  beta = 1 / (1 + b[2])
            else:
                  beta=1e-50
            x_new = ((1 - (beta * (b[0] * b[0]))) * point[0]) - ((beta * (b[0] * b[1])) * point[1]) - (b[0] * point[2])
            y_new = ((-beta * (b[0] * b[1])) * point[0]) + ((1 - (beta * (b[1] * b[1]))) * point[1]) - (b[1] * point[2])
            x_i = round((x_new + self.max_x) / self.dx)
            y_i = round((y_new + self.max_x) / self.dx)
            return np.array([b[0],b[1],b[2],x_i,y_i,(int(add)*2-1)])

      def updateVotingSpace(self,bxyv):
            bi=np.where(np.prod(np.equal(self.sphere.vertices,bxyv[:3]),axis=1))[0]
            self.VotingSpace[bi,int(bxyv[3]),int(bxyv[4])]=self.VotingSpace[bi,int(bxyv[3]),int(bxyv[4])]+bxyv[5]

      def pointVote(self,point,add):
            sub_pre_voting_space=np.vectorize(self.perVertexPointVote,signature="(k),(i),()->(j)")(point,self.sphere.vertices,add) # b x y vote ,n_vertices
            uniq=np.unique(sub_pre_voting_space,axis=0)
            sub_pre_voting_space=np.vectorize(lambda A,uniq_i: np.hstack((uniq_i,np.sum(A[np.prod(np.equal(A[:,:3],uniq_i[:3]),axis=1).astype(bool)][:,4]))),
                                          signature="(a,b),(i)->(j)")(sub_pre_voting_space,uniq)

            np.vectorize(self.updateVotingSpace,signature="(j)->()")(sub_pre_voting_space)


  # returns the line with most votes (rc = number of votes)
      def getLine(self):
            votes=np.max(self.VotingSpace)
            b_i,x_i,y_i=np.where(self.VotingSpace==np.max(self.VotingSpace))
            print(b_i)
            print(x_i)
            print(y_i)

            print("found "+str(b_i.shape[0])+" directions")
            if b_i.shape[0]>1:
                  #print("found "+str(b_i.shape[0])+" directions")
                  b_i=b_i[0]
                  x_i=x_i[0]
                  y_i=y_i[0]
            self.a=np.zeros((1,3))
            self.b=self.sphere.vertices[b_i,:].reshape(-1,)
            print(self.b)
            self.a[0,0] = x_i * (1 - ((self.b[0] * self.b[0]) / (1 + self.b[2]))) - y_i * ((self.b[0] * self.b[1]) / (1 + self.b[2]))
            self.a[0,1] = x_i * (-((self.b[0] * self.b[1]) / (1 + self.b[2]))) + y_i * (1 - ((self.b[1] * self.b[1]) / (1 + self.b[2])))
            self.a[0,2] = - x_i * self.b[0] - y_i * self.b[1]
            return votes

