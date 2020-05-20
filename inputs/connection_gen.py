attractorInhibit=[-1, 128]
shift=[3, 64]

for i in range(0, 8):
    if(i != 7):
        print(i, i+8, shift[0], shift[1])
    if(i != 0):
        print(i, i+14, shift[0], shift[1])
    for j in range(0, 8):
        if i != j:
            print(i, j, attractorInhibit[0], attractorInhibit[1])

for i in range(8, 15):
    print(i, i-7, shift[0], shift[1])
for i in range(15, 22):
    print(i, i-15, shift[0], shift[1])
