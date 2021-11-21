import Utilities

##Ejemplo fit
GR=3
P=6400
point=dp.points[P]
pt = o3d.geometry.KDTreeFlann(dp)
[k, idx, q] = pt.search_radius_vector_3d(point, 0.04)
zonal_points=np.asarray(dp.points)[idx[:], :]
[XY,Z]=transf(zonal_points,GR)
coe=fit(XY,Z)
vi=dp
vi.paint_uniform_color([0.8, 0.8, 0.8])
np.asarray(vi.colors)[idx[1:], :] = [0, 1, 0]
vi.estimate_normals()
o3d.visualization.draw_geometries([vi])

#def(ai,np)
#1) Formar ecuacion de superficie z=f(x,y)
x,y=sp.symbols('x y',real=True)
XY=genSymFun(x,y,GR)
z=XY*coe#--------------
z=z[0]
#2) Formar ecuacion parametrica r=(x,y,z(x,y)) ¿en matriz?

r=sp.matrices.Matrix([x,y,z])
#3) Determinar derivadas parciales ru,rv,ruu,ruv,rvv y 
    #normal (probar con normal estimada de nuble de puntos y de superficie fiteada)
ru=sp.diff(r,x)
rv=sp.diff(r,y)
ruu=sp.diff(ru,x)
ruv=sp.diff(ru,y)
rvv=sp.diff(rv,y)
ncp=dp.normals[200] #--------------------------
ns=ru.cross(rv)/(ru.cross(rv).norm())
#4) Determinar coeficientes L,M,M,E,F,G
#5) Formar matriz A y determinar autovalores y autovectores



#Graficar Espacio de curvaturas
p=0
r=60
print(query[1][p,:60])
fp=np.array(flatp)
plt.figure(figsize=(100,100))
#plt.plot(fp[:,0], fp[:,1],'bo',fp[query[1][p,:r],0], fp[query[1][p,:r],1],'ro')
plt.plot(fp[:,0], fp[:,1],'bo',fp[query[1][p,:],0], fp[query[1][p,:],1],'ro')
#plt.axis([0, 6, 0, 20])
plt.show()