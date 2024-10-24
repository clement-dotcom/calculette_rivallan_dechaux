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


def texte(self):
    texte1=input("entrez votre nombre: ")
    texte2=input("entre votre deuxieme nombre: ")




def addition(self,a,b):
    return a+b
def soustraction(self,a,b):
    return a-b
def multiplication(self,a,b):
    return a*b
def division(self,a,b):
    try:
        a//b
        return a/b
    except:
        print("division par zéro")



a=Bouton("+",100,350,40,'darkorange')
b=Bouton("-",200,350,40,'darkorange')
c=Bouton("x",300,350,40,'darkorange')
d=Bouton("/",400,350,40,'darkorange')
e=Bouton("=",20,220,40,'darkorange')

g.dessinerRectangle(20,20,460,80,"grey")
g.dessinerRectangle(20,120,460,80,"grey")

g.dessinerRectangle(240,220,200,80,"grey" )

g.attendreClic()
g.fermerFenetre()
