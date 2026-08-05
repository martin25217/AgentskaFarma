import numpy as np

x = int(input("Unsite broj stupaca prve matrice: "))
y = int(input("Unsite broj redaka prve matrice: "))
z = int(input("Unsite broj stupaca druge matrice: "))
f = int(input("Unsite broj redaka druge matrice: "))

matrix1 = []

for i in range(y):
    row = []
    for j in range(x):
        row.append(int(input("Unesite broj: ")))
    matrix1.append(row)



matrix2 = []

for i in range(f):
    row = []
    for j in range(z):
        row.append(int(input("Unesite broj: ")))
    matrix2.append(row)

matrix1 = np.array(matrix1)
matrix2 = np.array(matrix2)

print(matrix1)
print("\n")
print(matrix2)

if x != f:
    print("Matrice se ne mogu množiti!")
else:
    rezultat = np.dot(matrix1, matrix2)
    print("Rezultat je:")
    print(rezultat)