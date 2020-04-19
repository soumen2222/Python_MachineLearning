
total =0
count=0
average =0

while True :
    num = input("Enter a number")
    if num == 'done':
        break
    try:
        intnum = int(num)
        count =count +1
        total = total + intnum
    except:
        print("Invalid Input")

print("Total:",total, "count ", count , "Average" , (total/count))
