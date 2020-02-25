# Retrieve all data

def retrieve_aero_data():
  file="aerodynamicloadf100.dat"
  csv=open(file, "r").read()
  table = csv.split("\n")
  table=table[:-1]
  number_table=[]
  for element in table:
    number_table.append(eval(element))
  row=[]
  grid=[]
  for r in number_table:
    for e in r:
      row.append(float(e)*10**3)
    grid.append(row)
    row=[]
  return grid

#matrix with n rows, m columns
def matrix(n,m=0,e=0):
    if m==0:
        m=n
    A=[]
    for i in range(n):
        A.append([e]*m)
    return A

# turns list into vector
def vector(lst):
    n=len(lst)
    B=matrix(n,1)
    for i in range(n):
        B[i][0]=lst[i]
    return B

# transpose of a matrix
def transpose(A):
    if type(A[0])==int or type(A[0])==float:
        return vector(A)

    T=matrix(len(A[0]),len(A))
    for i in range(len(A)):
        for j in range(len(A[0])):
            T[j][i]=A[i][j]
    return T


aero_data = retrieve_aero_data()
grid = transpose(aero_data)
