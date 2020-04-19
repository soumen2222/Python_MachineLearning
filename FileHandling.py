fhand = open('C:\Data\SoumenPers\Python_WS\soumen.txt')

count =0
for file in fhand :
    count = count +1
print("Line Count :",count)

fhand1= open('C:\Data\SoumenPers\Python_WS\soumen.txt')
inp = fhand1.read()
print(len(inp))
print(inp[:20])

fhand2= open('C:\Data\SoumenPers\Python_WS\soumen.txt')


for file in fhand2 :

    if(file.startswith('Hi')) :
        print("Starts With" , file)


fname =input('Enter the file name: ')
try:
    fhand = open(fname)
except:
    print('File is not opened:', fname)
    quit()

count =0
for line in fhand :
    if(line.startswith('Hi')) :
        count = count +1
print('There were' , count)


# Use the file name mbox-short.txt as the file name
fname = input("Enter file name: ")
fh = open(fname)
count=0
Confidence=0
for line in fh:
    if not line.startswith("X-DSPAM-Confidence:") : continue
    count = count+1
    x=line.split(":")
    Confidence=Confidence+ float(x[1].lstrip())
print("Average spam confidence:",Confidence/count)



fname = input("Enter file name: ")
fh = open(fname)
lst = list()
lines =fh.read()
lines.strip()
x=lines.split(" ")
for i in x :
    if(i not in lst) :
        lst.append(i)
print(lst)