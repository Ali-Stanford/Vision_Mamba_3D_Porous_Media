//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	File: perm_SP_IFP.cpp
//
//	Implementation single-phase Lattice-Boltzmann Simulation with
//    (1) Fixed pressure boundary condition
//    (2) Implicit Matrix inversion with BiCGStab method.
//
//  Compiled binary name : permIFP.exe  
// 
//  Requred files: perm_SP_IFP.cpp (this file)
//                 lattice_SP_IFP.h (header file for method/classes)
//                 lattice_SP_IFP.cpp (implementation file)
//                 IntQueue.h (integer queue class, header)
//                 IntQueue.cpp (integer queue class, implementation)
//
//  Version History
//
//	Feb. 2003: ver. IFP 1.0.0 (initial implemenation)
//	Mar. 2003: ver. IFP 1.0.5 (add force term for better convergence) 
//	Mar. 2003: ver. IFP 1.1.0 (working version without the force term)
//
//	Copyright 2003 
//  Youngseuk Keehm, SRB Project, Stanford University
//  All rights reserved.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "lattice_SP_IFP.h"

void genDefault();
int  tryPamFile(char*);

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		genDefault();
		return 0;
	}
	
	if(tryPamFile(argv[1]))
	{
		Lattice lat(argv[1]);
		lat.calPerm();
	}
	return 0;
}

//////////////////////////////////////////////////////////////////////
// Function: tryPamFile()
//  Try to open the parameter file.
//	  return 0 : if the file does not exist    
//	  return 1 : if the file exists          
//              
//////////////////////////////////////////////////////////////////////
int tryPamFile(char* pamFile)
{
	std::ifstream fin(pamFile, std::ios::in);
	
	if(fin == 0)
	{
		std::cout << "\n Cannot find paramter file: " << pamFile << std::endl;
		std::cout << "Please check the file\n\n";
		return 0;
	}
	return 1;
}

//////////////////////////////////////////////////////////////////////
// Function: genDefault()
//  Shows the usage of this program
//	If the default parameter file (permIFP.pam) does not exist,
//	this function creates the sample parameter file.
//              
//////////////////////////////////////////////////////////////////////
void genDefault()
{
	std::cout << "\n Usage: permIFP pamfile\n\n";

	std::ifstream fin("permIFP.pam", std::ios::in);
	if(fin == 0){
		fin.close();
		std::cout << " Default parameter file is generated... permIFP.pam\n" << std::endl;

		std::ofstream fout("permIFP.pam");
		fout << "test.pore       :pore geometry filename\n";
		fout << "0               :no. of header lines (From Sisim : 3)\n";
		fout << "40 40 40        :nx ny nz\n";
		fout << "0.1             :dx (mm)\n";
		fout << "0               :write flux(0=no, 1=yes)\n";
		fout << "15              :buffer layers (15: default))\n";
		fout.close();
	}
	else
		fin.close();

}

// end of file