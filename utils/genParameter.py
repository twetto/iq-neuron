output = open("neuronParameter_Izhikevich_temp.txt","w+")
for i in range(0,1023+1):
    str="%d 40 70 80 1 1 3\n" %i
    output.write(str)
