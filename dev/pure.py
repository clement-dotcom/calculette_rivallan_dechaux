import tkiteasy as tk

X,Y=500,400
g = tk.ouvrirFenetre(X, Y)
on=True
nombre = {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', "ampersand": "1",
          "eacute": "2", "quotedbl": "3", 'apostrophe': "4", 'parenleft': "5", 'section': "6", 'egrave': '7',
          'exclam': '8', 'ccedilla': '9', 'agrave': '0'}
dict_operation={"+":"addition","-":"soustraction","x":"multiplication","/":"division"}
class Bouton:
    def __init__(self,type,x,y,r,color):
        self.type=type
        if type=="=":
            self.bouton=g.dessinerRectangle(x,y,200,80,color)
            self.txt=g.afficherTexte(type, x+100, y +30, "black", 60)

        else:

            self.bouton=g.dessinerDisque(x,y,r,color)
            self.txt=g.afficherTexte(type,x,y-5,"black",60)



def addition(a,b):
    return a+b
def soustraction(a,b):
    return a-b
def multiplication(a,b):
    return a*b
def division(a,b):
    try:
        a/b
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
    resultat=globals()[dict_operation[operation]](nb1, nb2)
    g.afficherTexte(resultat,350,250,"purple",30)


liste=initgraph()
listeA=[liste[i].bouton for i in range(4)]
listeB=[liste[i].txt for i in range(4)]
clickable=listeA+listeB
print(clickable)

text1,text2= "",""
clic=None
operation=""
texthaut, textbas = None, None

while on:
    touche = g.attendreTouche()
    if touche in nombre:
        text1 += nombre[touche]

        if texthaut is None:
            texthaut = g.afficherTexte(text1, 250, 70, "black", 50)

        else:
            g.changerTexte(texthaut, text1)

    elif clic := g.recupererClic():

        o = g.recupererObjet(clic.x, clic.y)

        for btn in liste[:-1]:
            if o == btn.bouton or o == btn.txt:
                operation = btn.type
                break

        if operation:
            break
    g.actualiser()

# DEUXIEME VARIABLE DE NOTRE CALCUL
deux=True
while deux :
    touche = g.attendreTouche()
    if touche in nombre:
        text2 += nombre[touche]

        if textbas is None:
            textbas = g.afficherTexte(text2, 250, 160, "black", 50)

        else:
            g.changerTexte(textbas, text2)

    elif clic := g.recupererClic():

        o = g.recupererObjet(clic.x, clic.y)

        if o == liste[-1].bouton or o == liste[-1].txt:
            resultat(int(text1), int(text2), operation)
            break


g.attendreClic()
g.fermerFenetre()


