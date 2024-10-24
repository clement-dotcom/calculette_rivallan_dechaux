import tkiteasy as tk
import time
X,Y=500,400
g=tk.ouvrirFenetre(X, Y)
g.dessinerRectangle(20,20,460,80,"grey")
g.dessinerRectangle(20,120,460,80,"grey")
g.dessinerRectangle(20,220,200,80,"darkorange" )
g.dessinerRectangle(240,220,200,80,"grey" )
a=g.dessinerDisque(100,350,40,'darkorange')
b=g.dessinerDisque(200,350,40,'darkorange')
c=g.dessinerDisque(300,350,40,'darkorange')
d=g.dessinerDisque(400,350,40,'darkorange')
plus=g.afficherTexte("+",100,345,"black",60)
moins=g.afficherTexte("-",200,345,"black",60)
multiplier=g.afficherTexte("x",300,345,"black",60)
diviser=g.afficherTexte("/",400,350,"black",60)
egal=g.afficherTexte("=",120,250,"black",150)

g.attendreClic()
g.fermerFenetre()


