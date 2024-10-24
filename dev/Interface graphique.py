import tkiteasy as tk
import time
X,Y=500,400
g=tk.ouvrirFenetre(X, Y)
class Bouton:
    def __init__(self,type,x,y,r,color):
        self.type=type
        if type=="=":
            self.bouton=g.dessinerRectangle(x,y,200,80,color)
            g.afficherTexte(type, x+100, y +30, "black", 60)

        else:

            self.bouton=g.dessinerDisque(x,y,r,color)
            g.afficherTexte(type,x,y-5,"black",60)

a=Bouton("+",100,350,40,'darkorange')
b=Bouton("-",200,350,40,'darkorange')
c=Bouton("x",300,350,40,'darkorange')
d=Bouton("/",400,350,40,'darkorange')
e=Bouton("=",20,220,40,'darkorange')



g.dessinerRectangle(20,20,460,80,"grey")
g.dessinerRectangle(20,120,460,80,"grey")

g.dessinerRectangle(240,220,200,80,"grey" )

"""""""""""
plus=g.afficherTexte("+",100,345,"black",60)
moins=g.afficherTexte("-",200,345,"black",60)
multiplier=g.afficherTexte("x",300,345,"black",60)
diviser=g.afficherTexte("/",400,350,"black",60)
egal=g.afficherTexte("=",120,250,"black",150)
"""""
g.attendreClic()
g.fermerFenetre()


