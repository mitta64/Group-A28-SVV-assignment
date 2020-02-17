def integral(f,x1,x2,res=10000):
    i=(x2-x1)/res   # interval
    A=0
    a=f(x1)
    for e in range(res):
        b=f(x1+(e+1)*i)
        A+=(a+b)*i/2
        a=b
    return A
