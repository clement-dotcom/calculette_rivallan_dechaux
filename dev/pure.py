import tkiteasy as tk
import time
X,Y=500,400
g = tk.ouvrirFenetre(X, Y)
on=True
nombre = {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', "ampersand": "1",
          "eacute": "2", "quotedbl": "3", 'apostrophe': "4", 'parenleft': "5", 'section': "6", 'egrave': '7',
          'exclam': '8', 'ccedilla': '9', 'agrave': '0'}

class Bouton:
    def __init__(self,type,x,y,r,color):
        self.type=type
        if type=="=":
            self.bouton=g.dessinerRectangle(x,y,200,80,color)
            self.txt=g.afficherTexte(type, x+100, y +30, "black", 60)

        else:

            self.bouton=g.dessinerDisque(x,y,r,color)
            self.txt=g.afficherTexte(type,x,y-5,"black",60)





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

def initgraph():

    a=Bouton("+",100,350,40,'darkorange')
    b=Bouton("-",200,350,40,'darkorange')
    c=Bouton("x",300,350,40,'darkorange')
    d=Bouton("/",400,350,40,'darkorange')
    e=Bouton("=",20,220,40,'darkorange')

    g.dessinerRectangle(20,20,460,80,"grey")
    g.dessinerRectangle(20,120,460,80,"grey")

    g.dessinerRectangle(240,220,200,80,"grey" )
    return [a,b,c,d,e]


def resultat(nb1,nb2,operation):
    g.afficherTexte("CDECHAUX",350,250,"purple",30)
    return None


liste=initgraph()

listeA=[liste[i].bouton for i in range(4)]
listeB=[liste[i].txt for i in range(4)]
clickable=listeA+listeB
print(clickable)



text1=""
text2=""
cpt1=0
clic=None

while on:
    clic = g.recupererClic()
    if clic != None:

        o = g.recupererObjet(clic.x, clic.y)

        if o in clickable:

            text2 = text1[-1]
            text1 = text1[:-1]
            textbas = g.afficherTexte(text2, 250, 160, "black", 50)
            g.changerTexte(texthaut, text1)


            for i in range(4):
                if o == liste[i].bouton or o == liste[i].txt:
                    operation = liste[i].type
            break

    if clic==None:


        touche=g.attendreTouche()

        if touche in nombre.keys():
            text1+=nombre[touche]

            if cpt1==0:
                texthaut=g.afficherTexte(text1,250,70,"black",50)
                cpt1=1
            else:
                g.changerTexte(texthaut,text1)


    g.actualiser()

deux=True
while deux :
    clic = g.recupererClic()
    if clic == None:

        touche = g.attendreTouche()

        if touche in nombre.keys():
            text2 += nombre[touche]

            g.changerTexte(textbas, text2)

    if clic!=None:

        o=g.recupererObjet(clic.x,clic.y)

        if o == liste[-1].bouton or o == liste[-1].txt:
            text2=text2[:-1]
            g.changerTexte(textbas,text2)
            print("fin")
            resultat(text1,text2,operation)



            deux=False


g.attendreClic()
g.fermerFenetre()


