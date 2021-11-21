from FreeCAD import Base
import Part,PartGui


Or=Base.Vector(0,0,3)
N=Base.Vector(0,1,1)
Pa=Part.makePlane(30,30,Or,N)
Part.show(Pa)

import PartDesignGui

App.activeDocument().addObject('Sketcher::SketchObject', 'CSkt')
App.activeDocument().CSkt.MapMode = "FlatFace"
App.activeDocument().CSkt.Support = [(App.getDocument('Unnamed').getObject('Shape'),'')]
#App.activeDocument().recompute()
ActiveSketch = App.getDocument('Unnamed').getObject('prueba')


Po=Part.makePolygon([(2,-3,0),(1,-3,0),(1,3,0),(2,3,0),(2,-3,0)])
Part.show(Po)

#PRUEBA 2
App.getDocument('Unnamed').getObject('CSkt').addGeometry(Part.LineSegment(App.Vector(25.174313,-21.116055,0),App.Vector(37.600494,-24.978247,0)),False)

#FOR
App.getDocument('Unnamed').getObject('CSkt').addGeometry(Part.LineSegment(App.Vector(37.600494,-24.978247,0),App.Vector(50.782322,-17.421784,0)),False)
App.getDocument('Unnamed').getObject('CSkt').addConstraint(Sketcher.Constraint('Coincident',0,2,1,1)) 

App.getDocument('Unnamed').getObject('CSkt').addGeometry(Part.LineSegment(App.Vector(50.782322,-17.421784,0),App.Vector(40.874962,-12.636024,0)),False)
App.getDocument('Unnamed').getObject('CSkt').addConstraint(Sketcher.Constraint('Coincident',1,2,2,1)) 

App.getDocument('Unnamed').getObject('CSkt').addGeometry(Part.LineSegment(App.Vector(40.874962,-12.636024,0),App.Vector(26.181841,-18.261391,0)),False)
App.getDocument('Unnamed').getObject('CSkt').addConstraint(Sketcher.Constraint('Coincident',2,2,3,1)) 

App.getDocument('Unnamed').getObject('CSkt').addGeometry(Part.LineSegment(App.Vector(26.181841,-18.261391,0),App.Vector(25.090353,-21.200015,0)),False)
App.getDocument('Unnamed').getObject('CSkt').addConstraint(Sketcher.Constraint('Coincident',3,2,4,1))

App.getDocument('Unnamed').getObject('CSkt').addConstraint(Sketcher.Constraint('Coincident',4,2,0,1)) 

#Extrucción
App.getDocument('Unnamed').addObject('Part::Extrusion','Extrude')
f = App.getDocument('Unnamed').getObject('Extrude')
f.Base = App.getDocument('Unnamed').getObject('CSkt')
f.DirMode = "Normal"
f.DirLink = None
f.LengthFwd = 10.000000000000000
f.LengthRev = 0.000000000000000
f.Solid = True
f.Reversed = False
f.Symmetric = False
f.TaperAngle = 0.000000000000000
f.TaperAngleRev = 0.000000000000000