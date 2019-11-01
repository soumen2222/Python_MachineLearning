stuff = dict()
print(stuff.get('candy',-1))

name = input("Enter file:")
if len(name) < 1: name = "mbox-short.txt"
handle = open(name)
dic = dict()
for line in handle:
    if not line.startswith("From "): continue
    x = line.split(" ")
    dic[x[1]] = dic.get(x[1], 0) + 1

max = 0
mail = ''
for a, b in dic.items():
    if (b > max):
        max = b
        mail = a

print(mail, max)