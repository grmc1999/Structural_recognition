import sys
import os
sys.path.append('D:\\Documentos\\INNOVATE\lib\\FreeCAD\\bin')
sys.path.append('D:\\Documentos\\INNOVATE\lib\\FreeCAD\\lib')
from FreeCAD import Base
import FreeCAD
import Part
import numpy as np
import pathlib
import getopt
#pth=str(pathlib.Path().absolute())

pth=os.path.dirname(os.path.realpath(__file__))


def GenTube(N,C,Re,l,e=2):
    N=Base.Vector(N)
    C=Base.Vector(C)
    Ri=Re-e
    Fe=Part.Face(Part.Wire(Part.Circle(C,N,Re).toShape()))
    Fi=Part.Face(Part.Wire(Part.Circle(C,N,Ri).toShape()))
    F=Fe.cut(Fi)
    Tu=F.extrude(N*l)
    #Part.show(Tu)
    return Tu

def read_file(path='\DataIn\geom_gen.txt'):
    pth=str(pathlib.Path().absolute())
    pth=os.path.dirname(os.path.realpath(__file__))
    #DEBUGGING
    #pth="D:\Documentos\INNOVATE\GH\proyectox\Generacion"
    txt=open(pth+path)
    body=txt.readlines()
    body=[i.split('\n')[0] for i in body]
    pipes=[]
    beams=[]
    planes=[]
    for i in body:
        if i.split(" ")[0]=="pipe":
            pipes.append(i)
        elif i.split(" ")[0]=="plane":
            planes.append(i)
        else:
            beams.append(i)
    return pipes,beams,planes

def parse_beam_to_dict(data):
    data=data.split(" ")
    data={'Type':data[0],
    'Xx':float(data[1]),'Xy':float(data[2]),'Xz':float(data[3]),
    'Yx':float(data[4]),'Yy':float(data[5]),'Yz':float(data[6]),
    'Zx':float(data[7]),'Zy':float(data[8]),'Zz':float(data[9]),
    'Px':float(data[10]),'Py':float(data[11]),'Pz':float(data[12]),
    'a':float(data[13]),
    'b':float(data[14]),
    'ea':float(data[15]),
    'eb':float(data[16]),
    'L':float(data[17]),
    'S':data[18]
    }
    return data

def parse_plane_to_dict(data):
    data=data.split(" ")
    data={'Type':data[0],
    'Xx':float(data[1]),'Xy':float(data[2]),'Xz':float(data[3]),
    'Yx':float(data[4]),'Yy':float(data[5]),'Yz':float(data[6]),
    'Zx':float(data[7]),'Zy':float(data[8]),'Zz':float(data[9]),
    'Px':float(data[10]),'Py':float(data[11]),'Pz':float(data[12]),
    'a':float(data[13]),
    'b':float(data[14]),
    'ea':float(data[15])
    }
    return data

def parse_pipe_to_dict(data):
    data=data.split(" ")
    data={'Type':data[0],
    'Zx':float(data[1]),'Zy':float(data[2]),'Zz':float(data[3]),
    'Px':float(data[4]),'Py':float(data[5]),'Pz':float(data[6]),
    'D':float(data[7]),
    'L':float(data[8]),
    'S':data[9]
    }
    return data

def draw_Plane_points(data):
    a=data["a"]
    b=data["b"]
    ea=data["ea"]

    points=np.array([
        [a/2,b/2,0],#1
        [a/2,-b/2,0],#2
        [-a/2,-b/2,0],#3
        [-a/2,b/2,0],#4
    ])
    return points

def draw_W_points(data):
    a=data["a"]
    b=data["b"]
    ea=data["ea"]
    eb=data["eb"]

    points=np.array([
        [a/2,-b/2,0],#1
        [a/2,b/2,0],#2
        [(a/2-eb),b/2,0],#3
        [(a/2-eb),ea/2,0],#4
        [-(a/2-eb),ea/2,0],#5
        [-(a/2-eb),b/2,0],#6
        [-a/2,b/2,0],#7
        [-a/2,-b/2,0],#8
        [-(a/2-eb),-b/2,0],#9
        [-(a/2-eb),-ea/2,0],#10
        [(a/2-eb),-ea/2,0],#11
        [(a/2-eb),-b/2,0],#12
    ])
    return points

def draw_T_points(data):
    a=data["a"]
    b=data["b"]
    ea=data["ea"]
    eb=data["eb"]

    points=np.array([
        [-a/2,-b/2,0],#1
        [-a/2,b/2,0],#2
        [-(a/2-eb),b/2,0],#3
        [-(a/2-eb),ea/2,0],#4
        [a/2,ea/2,0],#5
        [a/2,-ea/2,0],#6
        [-(a/2-eb),-ea/2,0],#7
        [-(a/2-eb),-b/2,0]#8
    ])
    return points

def draw_L_points(data):
    a=data["a"]
    b=data["b"]
    ea=data["ea"]
    eb=data["eb"]

    points=np.array([
        [-a/2,b/2,0],#1
        [-a/2,-b/2,0],#2
        [a/2,-b/2,0],#3
        [a/2,-b/2+ea,0],#4
        [-a/2+eb,-b/2+ea,0],#5
        [-a/2+eb,b/2,0]#6
    ])
    return points

def draw_C_points(data):
    a=data["a"]
    b=data["b"]
    ea=data["ea"]
    eb=data["eb"]

    points=np.array([
        [-a/2,-b/2,0],#1
        [-a/2,b/2,0],#2
        [-a/2+eb,b/2,0],#3
        [-a/2+eb,-b/2+ea,0],#4
        [a/2-eb,-b/2+ea,0],#5
        [a/2-eb,b/2,0],#6
        [a/2,b/2,0],#7
        [a/2,-b/2,0]#8
    ])
    return points


def pointsToFace(points):
    vec_points=np.vectorize(lambda point: Base.Vector(point[0],point[1],point[2]) ,otypes=[object],signature="(i)->()")(points)

    n_points=points.shape[0]

    lines=np.vectorize(lambda vec_id,vec: Part.LineSegment(vec[vec_id%n_points], vec[(vec_id+1)%n_points]),
        otypes=[object],
        signature="(),(j)->()" )(np.arange(0,n_points),vec_points)

    Shape=Part.Shape(lines)
    Wire=Part.Wire(Shape.Edges)
    Face=Part.Face(Wire)
    return Face


def transform_profile(face,data):
    A0=np.eye(3)
    A1=np.array([
        [data['Xx'],data['Yx'],data['Zx']],
        [data['Xy'],data['Yy'],data['Zy']],
        [data['Xz'],data['Yz'],data['Zz']]
    ])
    A1=A1/np.sum(A1**2,axis=0)**0.5
    R=np.matmul(A1,np.linalg.inv(A0))
    p=np.array([
        [data['Px']],
        [data['Py']],
        [data['Pz']]
    ])
    A=np.vstack((np.hstack((R,p)),np.array([[0,0,0,1]])))
    At=Base.Matrix()
    At.A11=A[1-1,1-1]
    At.A12=A[1-1,2-1]
    At.A13=A[1-1,3-1]
    At.A14=A[1-1,4-1]
    At.A21=A[2-1,1-1]
    At.A22=A[2-1,2-1]
    At.A23=A[2-1,3-1]
    At.A24=A[2-1,4-1]
    At.A31=A[3-1,1-1]
    At.A32=A[3-1,2-1]
    At.A33=A[3-1,3-1]
    At.A34=A[3-1,4-1]
    At.A41=A[4-1,1-1]
    At.A42=A[4-1,2-1]
    At.A43=A[4-1,3-1]
    At.A44=A[4-1,4-1]
    face=face.transformGeometry(At)
    return face

def extrude_profile(lines,data):
    Shape=Part.Shape(lines)
    Wire=Part.Wire(Shape.Edges)
    F=Part.Face(Wire)
    Nd=np.array([data['Zx'],data['Zy'],data['Zz']])
    Nd=Nd/np.linalg.norm(Nd)
    N=Base.Vector(Nd*data['L'])
    P=F.extrude(N)
    return P

def extrude_plane(lines,data):
    Shape=Part.Shape(lines)
    Wire=Part.Wire(Shape.Edges)
    F=Part.Face(Wire)
    Nd=np.array([data['Zx'],data['Zy'],data['Zz']])
    Nd=Nd/np.linalg.norm(Nd)
    N=Base.Vector(Nd*data['ea'])
    P=F.extrude(N)
    return P

def pipeline(i_path,o_path):
    pth=str(pathlib.Path().absolute())
    #DEBUGGING
    #pth="D:\Documentos\INNOVATE\GH\proyectox\Generacion"
    Name=o_path
    #Dest=pth.split('\\')[:-1]
    Dest=pth.split('\\')
    Dest='\\'.join(Dest)+'\\'+Name+'.iges'
    Doc=FreeCAD.newDocument(Name)

    pipes,beams,planes=read_file(i_path)

    solid_bodys=[]

    for pipe in pipes:
        body_data=parse_pipe_to_dict(pipe)

        #DEBUGGING
        #print(body_data['S'])

        N=np.array([body_data['Zx'],body_data['Zy'],body_data['Zz']])
        N=N/np.linalg.norm(N)
        C=np.array([body_data['Px'],body_data['Py'],body_data['Pz']])
        Re=body_data['D']/2
        l=body_data['L']
        Solid=GenTube(N,C,Re,l)

        Part.show(Solid)

        solid_bodys.append(Solid)

    for plane in planes:

        body_data=parse_plane_to_dict(plane)

        points=draw_Plane_points(body_data)
        Face=pointsToFace(points)

        TFace=transform_profile(Face,body_data)
        Solid=extrude_plane(TFace,body_data)
        solid_bodys.append(Solid)

        Part.show(Solid)
    
    for beam in beams:

        body_data=parse_beam_to_dict(beam)

        #DEBUGGING
        #print(body_data['S'])

        if body_data['Type']=="W":
            points=draw_W_points(body_data)
            Face=pointsToFace(points)

        elif body_data['Type']=="T":
            points=draw_T_points(body_data)
            Face=pointsToFace(points)

        elif body_data['Type']=="L":
            points=draw_L_points(body_data)
            Face=pointsToFace(points)

        elif body_data['Type']=="C":
            points=draw_C_points(body_data)
            Face=pointsToFace(points)

        else:
            print("type not valid")

        TFace=transform_profile(Face,body_data)
        Solid=extrude_profile(TFace,body_data)
        solid_bodys.append(Solid)

        Part.show(Solid)

    GenTubes=Doc.Objects
    Part.export(GenTubes,Dest)

    return Face,TFace,Solid




def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print(inputfile)
    print(outputfile)
    pipeline(inputfile,outputfile)
    

if __name__ == "__main__":
   main(sys.argv[1:])


#COMANDO
#python .\Generacion_vigas.py -i "\DataIn\geom_gen.txt" -o "prueba"