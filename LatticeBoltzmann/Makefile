all: perm_SP_IFP clean

perm_SP_IFP: lattice_SP_IFP.o IntQueue.o perm_SP_IFP.o
	g++ lattice_SP_IFP.o IntQueue.o perm_SP_IFP.o -o perm_SP_IFP

lattice_SP_IFP.o: source/lattice_SP_IFP.cpp
	g++ -c source/lattice_SP_IFP.cpp

IntQueue.o: source/IntQueue.cpp
	g++ -c source/IntQueue.cpp

perm_SP_IFP.o: source/perm_SP_IFP.cpp
	g++ -c source/perm_SP_IFP.cpp

clean:
	rm *o 
