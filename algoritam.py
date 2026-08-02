import random 
opulacija
numeriraj populacija
bodovanje
izdvojimo 30%
random.choice(populacija)
class Jedinka:
    def __init__(self, ime, rezultat):
        self.ime = ime
        self.rezultat = rezultat

    def __repr__(self):
        return f"{self.ime}: {self.rezultat}"

populacija = [Jedinka("A", 45), Jedinka("B", 92), Jedinka("C", 78)]

# Sortiranje na licu mjesta (od najvećeg do najmanjeg)
populacija.sort(key=lambda j: j.rezultat, reverse=True)
