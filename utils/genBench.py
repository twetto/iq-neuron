output = open("testbench_IQIF.txt","w+")
output.write("2000\n")
for i in range(0,127+1):
    str="%d 4\n" %(i)
    output.write(str)
output.write("-1\n")
output.write("1000\n")
for i in range(0,127+1):
    str="%d 0\n" %(i)
    output.write(str)
output.write("-1\n")
output.write("-1\n")
