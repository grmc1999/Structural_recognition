import sys
sys.path.append('D:\\Documentos\\INNOVATE\lib\\FreeCAD\\bin')
sys.path.append('D:\\Documentos\\INNOVATE\lib\\FreeCAD\\lib')
import FreeCAD
from FreeCAD import Base
import Part
import numpy as np

def GenTube(N,C,Re,e=2,l=20):
    N=Base.Vector(N)
    C=Base.Vector(C)
    Ri=Re-e
    Fe=Part.Face(Part.Wire(Part.Circle(C,N,Re).toShape()))
    Fi=Part.Face(Part.Wire(Part.Circle(C,N,Ri).toShape()))
    F=Fe.cut(Fi)
    Tu=F.extrude(N*l)
    Part.show(Tu)
    return Tu

def mainProcess(path="D:\\Documentos\\INNOVATE\\GH\\proyectox\\Generación\\DataIn\\inference_pred.txt"):

    Name=path.split('\\')[-1].split('_')[0]
    Dest=path.split('\\')[:-1]
    Dest='\\'.join(Dest)+'\\'+Name+'.iges'
    Doc=FreeCAD.newDocument(Name)
    txt=open(path)
    Tubes=txt.readlines()
    Tubes=[i.split('\n')[0].split(' ')[:-1] for i in Tubes]
    Tubes=np.asarray(Tubes,dtype=np.float64)
    GenTubes=[]

    for Tube in Tubes:
        N=Tube[:3]
        N=-N/np.linalg.norm(N)
        C=Tube[3:6]
        R=Tube[6]*25.4/2
        e=Tube[6]/10
        l=Tube[7]
        GenTubes.append(GenTube(N,C,R,e,l))

    GenTubes=Doc.Objects

    Part.export(GenTubes,Dest)

mainProcess()