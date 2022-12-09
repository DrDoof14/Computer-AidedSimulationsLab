import re

divina_commedia = open("divina_commedia.txt", "r", encoding = "UTF-8").read()
divina_commedia = re.sub(r'[^\w\s]', '', divina_commedia)

## PREPROCESSING
dc_a_capo = divina_commedia.split("\n")
# removing titles
del dc_a_capo[0:8]

num_verses = len(dc_a_capo)

dc_parole = []
for line in dc_a_capo:
    if line.startswith("Inferno") or line.startswith("Purgatorio") or line.startswith("Paradiso"):
        dc_a_capo.remove(line)
    else:
        linea_splittata = line.split(" ")
        for parola in linea_splittata:
            if parola == "":
                linea_splittata.remove(parola)
        dc_parole.extend(linea_splittata)


num_words = len(dc_parole)
dist_num_words = len(set(dc_parole))

sentence = input()  
sentence = re.sub(r'[^\w\s]', '', sentence)
sentence = sentence.split(" ")


