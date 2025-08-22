//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	File: lattice_SP_IFP.h
//
//	Implementation of single-phase Lattice-Boltzmann Simulation with
//    (1) Fixed pressure boundary condition
//    (2) Implicit Matrix inversion with BiCGStab method.
//
//  Compiled binary name : permIFP.exe  
// 
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

#ifndef _lattice_SP_IFP
#define _lattice_SP_IFP


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "IntQueue.h"

#define L_Debug		0				// debugging switch (0-no, 1-yes)
#define NCHAR		128
#define NCLUSTER	100 

class Lattice{

public:
	Lattice(char *pamFileName);	// constuctor
	~Lattice();			 		// destructor
	int calPerm();				// initialize density distribution and propagation mapping


private:

/*  physical parameters*/
	double dPdx, density, DP;	// pressure gradient, density
	double nu, viscosity;		// viscosity
	double porosity, totPerm;	// (calculated) porosity and perm
	double perm;				// perm for each cluster

/* pore structure parameters */
	int    nx, ny, nz, nxT; 	// size of digital rock
	int    ELayers;
	int    nInlet, nOutlet;
	double *IForce, *OForce;
	double dx;					// grid spacing (mm)
	int    nNodes;				// number of pore nodes in the rock
	int	   numClusters;			// number of clusters
	int	   nClustElem[NCLUSTER];// number of nodes at each cluster
	char   clustIndex[NCLUSTER];// index of each cluster (start with 'A')

/* filenames */
	char *poreFileName;			// input filename for pore geometry
	char *resFileName;			// result filename
	char *fluxFileName;			// flux field output filename

/* control parameter */
	int    noHeaderLines;		// number of headers in pore geometry file
	int    dumpInterval;		// interval for evaluating convergence
	double tolerance;			// convergence criteria
	double tol_norm, tol_perm;
	int    maxIter;				// maximum number of iterations
	bool   writeFlux;			// true for writing flux file
	bool   periodicBC;			// true for periodic BC condition (Y&Z directions)
	int    mirroring;			// 0 for not mirror the structure (already periodic)

/* Lattice parameters */
	double *tmp1, *tmp2;		// temp. arrays for A*x operation
	double P[4][18];			// Projection matrix 
	double E[18][4];			// Expansion matrix
	double F[18];				// Force vector
	int    C[18][3];			// Velocity vectors
	int    np;					// number of velocity vectors (18)
	char   ***type;				// pore geometry (0-pore, otherwise-grains)
	int    *updates;			// indices for propagation mapping
	double Q[4];				// volume-averaged flux
	double cs2, a01, a02;		// constant parameters for lattice collision


/* Private member funtion */
	void readParameters(char*);	// reading parameters
	void readPores();			// reading pore geometry
	int  clustering();			// cluster anaylsis
	void initNodes(char, int, double*);	// set up propagation indices

	// BiCGStab Solver and Mat-Vec operators
	int    solve  (int, double*, double*);
	void   MatVec (int, double*, double*);
	void   VecAXPY(int, double,  double*, double*, double*);
	double VecNorm(int, double*);
	double VecDot (int, double*, double*);
	void   VecCopy(int, double*, double*);
	void   VecSet (int, double,  double*);
	void   calRHS (int, double*);
	
	void calCurrentFlux(int, double*);	// compute total averaged flux	
	
};


#endif

// end of file
