x =5
if x < 2 :
    print("small")
elif x > 10 :
    print("bigger")
else :
    print("number 5")

rawstar = input("Enter the Raw number")

try:
    ival = int(rawstar)
except:
    ival =-1

if(ival>0) :
    print("Nice Work")
else :
    print("Not a number")

