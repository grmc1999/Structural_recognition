import FreeCAD,Mesh
import Part, os
import CompoundTools.Explode

#todos los archivos

path="D:\Documentos\INNOVATE\Base_de_datos\Para envio\LINEA 12\LINEA_12_IGES"
dir=os.listdir(path)
ptho=path+"_STL"
os.makedirs(ptho, exist_ok=True)
txt=open(ptho+"\\"+"CM.txt","a")

for f in dir:
	nn=f.replace(' ','_')
	nn=f.replace('#','_')
	os.rename(path+'\\'+f,path+'\\'+nn)
dir=os.listdir(path)
#try:
for f in dir:
    p=path+'\\'+f
    Part.insert(p,f.split('.')[0])
    cd=FreeCAD.getDocument(f.split('.')[0])
    objs=cd.Objects
    for i in range(len(objs)):
        print(ptho+'\\'+f.split('.')[0]+".stl")
        if(objs[i].Shape.ShapeType=='Compound'):
            CompoundTools.Explode.explodeCompound(objs[i])

    objs=cd.Objects
    for i in range(len(objs)):
        if("child" in objs[i].Name):
            print(objs[i].Name)
            Mesh.export([objs[i]],ptho+'\\'+f.split('.')[0]+objs[i].Name+".stl")
            S=f.split('.')[0]+objs[i].Name+" "+str(objs[i].Shape.BoundBox.XMax)+" "+\
                str(objs[i].Shape.BoundBox.YMax)+" "+str(objs[i].Shape.BoundBox.ZMax)+\
                    " "+str(objs[i].Shape.BoundBox.XMin)+" "+str(objs[i].Shape.BoundBox.YMin)+\
                        " "+str(objs[i].Shape.BoundBox.ZMin)+"\n"
            txt.write(S)

    name=f.split('.')[0]
    App.closeDocument(name)
txt.close()
