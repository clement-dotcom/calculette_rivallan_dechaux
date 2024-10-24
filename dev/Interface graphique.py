import tkiteasy as tk
import time
X,Y=500,400
g=tk.ouvrirFenetre(X, Y)
g.dessinerRectangle(20,20,460,80,"white")
g.dessinerRectangle(20,120,460,80,"pink")
g.dessinerRectangle(20,220,200,80,"white" )
g.dessinerRectangle(240,220,200,80,"blue" )
a=g.dessinerDisque(100,350,40,'orange')
b=g.dessinerDisque(200,350,40,'orange')
c=g.dessinerDisque(300,350,40,'orange')
d=g.dessinerDisque(400,350,40,'orange')
plus=g.afficherTexte()

g.attendreClic()
g.fermerFenetre()


