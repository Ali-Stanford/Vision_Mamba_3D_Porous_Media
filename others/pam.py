import operator

data_number = 5000
nx = 64
ny = 64
nz = 64
dx = 0.003

for i in range(data_number):   
    with open(str(1+i)+'.pam', 'w') as f:
        f.write('3Dporous'+str(1+i)+'.dat'+'\n')
        f.write('0\n')
        f.write(str(nx)+' '+str(ny)+' '+str(nz)+'\n')
        f.write(str(dx)+'\n')
        f.write('1\n')
        f.write('15')
    f.close()       
