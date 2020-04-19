fname = input("Enter file name: ")
fh = open(fname)
lst = list()
newfile=fh.read()
newfile.strip()
x= newfile.splitlines( )
for line in x :
    y = line.split(' ')
    for word in y :
        if(word not in lst) :
            lst.append(word)
lst.sort()
print(lst)

fname = input("Enter file name: ")
if len(fname) < 1: fname = "mbox-short.txt"

fh = open(fname)
count = 0
lst = list()

for line in fh:
    if not line.startswith("From "): continue
    count = count + 1
    x = line.split(" ")
    lst.append(x[1])

for i in lst:
    print(i)

print("There were", count, "lines in the file with From as the first word")




