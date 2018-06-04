

target = [[0,10,0,3],[3,5,0,7],[3,0,0,2],[0,0,2,0],[0,0,0,5]]
bin = [[0,1,0,1],[1,1,0,1],[1,0,0,1],[0,0,1,0],[0,0,0,1]]

b = [[2,20,44,5],[2,5,2,7],[0,1,0,9],[1,2,1,0],[1,0,0,8]]


def get_bin(lol):
    bin_lol = []
    
    for i in range(len(lol)):
        bin_l = [0] * len(lol[0])
        for j in range(len(lol[0])):
            if lol[i][j] != 0:
                bin_l[j] = 1

        bin_lol.append(bin_l)
    return bin_lol
    
blah = get_bin(target)
print(blah)
for i in range(len(b)):
    for j in range(len(b[0])):
        print(( blah[i][j] * (b[i][j] - target[i][j])**2) / len(b[0]))

