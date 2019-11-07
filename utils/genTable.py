output = open("Connection_Table_temp.txt","w+")
for i in range(0,1023+1):
    for j in range(i-384,i+384):
        if i != j and i >= 0 and i <= 1023 and j >= 0 and j <= 1023:
            str="%d %d 2 100\n" %(i,j)
            output.write(str)
