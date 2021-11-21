import sys
from ...libreria.Primerav import *

pt = Parser(path="D:\\Abdigal\\Torre\\Torre 02\\T02.pts", minv=True)
mesh = o3d.io.read_triangle_mesh("D:\\Abdigal\\Torre\\Torre 02\\total.stl")
o3d.visualization.draw_geometries([pt])