//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	File: lattice_SP_IFP.cpp
//
//	Implementation single-phase Lattice-Boltzmann Simulation with
//    (1) Fixed pressure boundary condition
//    (2) Implicit Matrix inversion with BiCGStab method.
//
//  Compiled binary name : permIFP.exe  
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

//////////////////////////////////////////////////////////////////////
// Function: Lattice()
//  Constuctor
//	Input		: pamFileName (Parameter file name)
//	Variables	: np (number of velocity vectors)
//				  nu (viscosity of fluid: 1/6)
//				  cs2 (square of speed of sound)
//				  a01 (lattice coefficient r=1)
//				  a02 (lattice coefficient r=sqrt(2))
//                C (velocity vector)
//                E (Expansion Matrix- from rho, q -> n_i)
//                P (projection matrix- from n_i --> rho, q
//              
//////////////////////////////////////////////////////////////////////
Lattice::Lattice(char *pamFileName)
{
	int i;

	np			 = 18;					
	density		 = 1.0;
	nu			 = density/6.0;	
	cs2			 = 0.5;

	DP           = 1e-3;
	tol_norm	 = 1e-6;
	tol_perm	 = 5e-5;
	dumpInterval = 10;
	maxIter		 = 10000;


	// setup velocity vector
	C[0][0]  = 1;	C[0][1]  = 0; 	C[0][2]  = 0;
    C[1][0]  =-1;	C[1][1]  = 0; 	C[1][2]  = 0;
    C[2][0]  = 0;	C[2][1]  = 1; 	C[2][2]  = 0;
    C[3][0]  = 0;	C[3][1]  =-1; 	C[3][2]  = 0;
    C[4][0]  = 0;	C[4][1]  = 0; 	C[4][2]  = 1;
    C[5][0]  = 0;	C[5][1]  = 0; 	C[5][2]  =-1;

    C[6][0]  = 1;	C[6][1]  = 1; 	C[6][2]  = 0;
    C[7][0]  =-1;	C[7][1]  =-1; 	C[7][2]  = 0;
    C[8][0]  = 1;	C[8][1]  =-1; 	C[8][2]  = 0;
    C[9][0]  =-1;	C[9][1]  = 1; 	C[9][2]  = 0;

    C[10][0] = 1;	C[10][1] = 0; 	C[10][2] = 1;
    C[11][0] =-1;	C[11][1] = 0; 	C[11][2] =-1;
    C[12][0] = 1;	C[12][1] = 0; 	C[12][2] =-1;
    C[13][0] =-1;	C[13][1] = 0; 	C[13][2] = 1;

    C[14][0] = 0;	C[14][1] = 1; 	C[14][2] = 1;
    C[15][0] = 0;	C[15][1] =-1;	C[15][2] =-1;
    C[16][0] = 0;	C[16][1] = 1; 	C[16][2] =-1; 
    C[17][0] = 0;	C[17][1] =-1;	C[17][2] = 1;

	// setup expansion and projection matrices
	// When cs (square of sound speed) = 0.5, collision coeffients are
	// a01 = 1/12, a02 = 1/24
	// a11 = 1/6,  a12 = 1/12, (see Ladd, 1994 for details)
	for(i=0; i<6; i++)
	{
		E[i][0] = 1.0/12.0;
		E[i][1] = C[i][0]/6.0;
		E[i][2] = C[i][1]/6.0;
		E[i][3] = C[i][2]/6.0;

		P[0][i] = 1.0;
		P[1][i] = C[i][0];
		P[2][i] = C[i][1];
		P[3][i] = C[i][2];
	}
	for(i=6; i<np; i++)
	{
		E[i][0] = 1.0/24.0;
		E[i][1] = C[i][0]/12.0;
		E[i][2] = C[i][1]/12.0;
		E[i][3] = C[i][2]/12.0;

		P[0][i] = 1.0;
		P[1][i] = C[i][0];
		P[2][i] = C[i][1];
		P[3][i] = C[i][2];
	}

	readParameters(pamFileName);
	readPores();
}

//////////////////////////////////////////////////////////////////////
// Function: ~Lattice()
// Destructor
//	Clear dynamically allocated memory
//////////////////////////////////////////////////////////////////////
Lattice::~Lattice()
{
	int i,j;
	
	// destroy 1D array
	delete [] poreFileName;
	delete [] resFileName;
	delete [] fluxFileName;


	for(i=0 ; i<nx ; i++)
	{
		for(j=0 ; j<ny ; j++)
			delete type[i][j];
		
		delete [] type[i];
	}
	delete [] type;

}

//////////////////////////////////////////////////////////////////////
// Function: readParameters(char*)
//	read parameters for the pore structure and
//  control parameters for flow simulation
//////////////////////////////////////////////////////////////////////
void Lattice::readParameters(char *pamFileName)
{
	char filename[NCHAR], dummy[NCHAR+1];
	int  dummyInt;
	
	std::ifstream fin(pamFileName);
	
	fin >> filename;		fin.getline(dummy, NCHAR);
	fin >> noHeaderLines;	fin.getline(dummy, NCHAR);
	fin >> nx >> ny >> nz;	fin.getline(dummy, NCHAR);
	fin >> dx;				fin.getline(dummy, NCHAR);
	fin >> dummyInt;		fin.getline(dummy, NCHAR);
		
	if(dummyInt == 0) writeFlux = false;
	else			  writeFlux = true;		

	fin >> ELayers;
	
	fin.close();


	poreFileName = new char[strlen(filename) + 1];
	resFileName  = new char[strlen(filename) + 5];
	fluxFileName = new char[strlen(filename) + 5];
	strcpy(poreFileName, filename);
	strcpy(resFileName, filename);
	strcpy(fluxFileName, filename);
	strcat(resFileName, ".res");
	strcat(fluxFileName, ".flx");
}

//////////////////////////////////////////////////////////////////////
// Function: readPores()
//	read indices for the pore geometry
//	(0-pore ; otherwise-grains)
//////////////////////////////////////////////////////////////////////
void Lattice::readPores()
{
	std::ifstream fin(poreFileName, std::ios::in);
	
	if(fin == 0)
	{
		std::cout << "\n The pore geometry file (" << poreFileName <<
			") does not exist!!!!\n";
		std::cout << " Please check the file\n\n";

		exit(0);
	}

	double pore;
	char dummy[NCHAR+1];
	int i, j, k, nx2;

	// Allocate memory for pore geometry (char***)

	nx2 = nx + 2*ELayers;
	
	type = new char**[nx2];
	for(i=0 ; i<nx2 ; i++){
		type[i] = new char*[ny];
		for(j=0 ; j<ny ; j++){
			type[i][j] = new char[nz];
		}
	}

	// Skip the headers (from sisim, the header is 3 lines)
	for(i=0 ; i<noHeaderLines ; i++)
		fin.getline(dummy, NCHAR);

	// Reading pore geometry
	nNodes = 0;
	for(k=0       ; k<nz         ; k++)
	for(j=0       ; j<ny         ; j++)
	for(i=ELayers ; i<nx+ELayers ; i++)
	{
		while(true)
		{	
			fin >> pore;
			if( pore == 0.0 || pore == 1.0) break;
		}
		if(pore == 0.0) {
			type[i][j][k] = '0';
			nNodes++;
		}
		else{
			type[i][j][k] = '1';
		}
	}
	fin.close();

	porosity = (double)nNodes / (nx*ny*nz);

	// Generate buffer zones
	for(i=0 ; i<ELayers ; i++)
	for(j=0 ; j<ny		; j++)
	for(k=0 ; k<nz		; k++)
	{
		type[i][j][k]            = type[ELayers][j][k];
		type[i+nx+ELayers][j][k] = type[nx+ELayers-1][j][k];
	}
	nx = nx2;

	// Assign Inlet and Outlet density
	//  which simulate fixed inlet/outlet pressures

	IForce = new double[np];
	OForce = new double[np];
	for(i=0 ; i<6 ; i++)
	{
		IForce[i] = (1.0+DP)/12.0;
		OForce[i] = 1.0/12.0;
	}
	for(i=6 ; i<np ; i++)
	{
		IForce[i] = (1.0+DP)/24.0;
		OForce[i] = 1.0/24.0;
	}	
}


//////////////////////////////////////////////////////////////////////
// Function: calPerm()
//	main routine for calculate permeability
//////////////////////////////////////////////////////////////////////
int Lattice::calPerm()
{
	int      i, j, k, nCNodes;
	int      n1, n2, nxx = nx - 2*ELayers;
	int      errCode, min, sec;
	char     flxFiles[NCHAR], buffer[10];
	time_t   stime, etime;
	std::ofstream fout;

	time(&stime);

	int nC = clustering();


	fout << "%=========================================================\n";
	fout << "% LB Single-phase Flow Simulator Ver.IFP.1.1.0 - Mar. 2003\n";
	fout << "%                   Youngseuk Keehm - All rights reserved.\n";
	fout << "%---------------------------------------------------------\n";
	fout << "% Pore filename  : " << poreFileName << " (" << nxx << " " 
								  << ny << " " << nz << ")\n"; 
	fout << "% Porosity       : " << porosity << "\n";
	fout << "% Grid spacing   : " << dx << "\n";
	fout << "% Extra Layers   : " << ELayers << "\n";
	fout << "% No. of Clusters: " << nC << "\n";
	fout << "%=========================================================" << std::endl;


	// No cluster for fluid to flow 
	if(nC == 0) {
		fout << "%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
		fout << "% Critical error - check the pore geometry\n";
		fout << "% No effective path for fluid to flow!!!\n" << std::endl;
		fout << "%=========================================================\n";
		

		fout.open(resFileName);

		fout << "%=========================================================\n";
		fout << "% LB Single-phase Flow Simulator Ver.IFP.1.1.0 - Mar. 2003\n";
		fout << "%                   Youngseuk Keehm - All rights reserved.\n";
		fout << "%---------------------------------------------------------\n";
		fout << "% Pore filename  : " << poreFileName << " (" << nxx << " " 
								  << ny << " " << nz << ")\n"; 
		fout << "% Porosity       : " << porosity << std::endl;
		fout << "% Grid spacing   : " << dx << "\n";
		fout << "% Extra Layers   : " << ELayers << "\n";
		fout << "% No. of Clusters: " << nC << "\n";
		fout << "%=========================================================\n";
		fout << "% Cal. Perm(mD)  : 0 \n";
		fout << "% No Connected Path \n";
		fout << "%=========================================================\n";

		fout.close();
		return -99;
	}

	// Calculate perm for each cluster

	// declare vectors
	double *x, *b;

	totPerm = 0.0;
	
	for(i=0 ; i<nC ; i++)				// loop for each independent cluster
	{
		nCNodes = nClustElem[i];

		// memory allocations for propagation indices
		x       = new double[ nCNodes*4 ];
		updates = new int[ nCNodes*np ];

		// calculate propagation indices
		initNodes(clustIndex[i], nCNodes, x);

		// memory allocations for Ax=b and temp. variables
		b    = new double[ nCNodes * 4 ];
		tmp1 = new double[ nCNodes * np + 2];

		// initialize b vector
		calRHS(nCNodes, b);

		// solve Ax = b		
		
		errCode = solve(nCNodes, x, b);
		
		// end of main loop
		
		if(errCode < 0) break;

		if(writeFlux)
		{
			strcpy(flxFiles, fluxFileName);
			strcat(flxFiles, ".");
			//itoa(i, buffer, 10);
			sprintf(buffer, "%d", i);
			strcat(flxFiles, buffer);
		
			n1 =  ELayers*nInlet;		
			n2 =  nCNodes -  ELayers*nOutlet;

			fout.open(flxFiles);
			for(j=n1*4; j<n2*4 ; j+=4)
				fout << x[j] << " " << x[j+1] << " " << x[j+2] << " " << x[j+3] << "\n";

			fout.close();
		}
		delete [] tmp1;
		delete [] x;
		delete [] b;
		delete [] updates;
	}

	if(errCode < 0) 
	{
		fout << "%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
		fout << "% Critical error at " << i << " Cluster in BiCGStab\n";
		fout << "% Error code reported: " << errCode << "\n";
		fout << "% Abnormal termination (LBSPFS_IFP.1.1.0)\n";
		fout << "%=========================================================\n";
		return errCode;
	}

	totPerm += perm;

	time(&etime);
	min = (int)( (etime-stime)/60 );
	sec = (int)(etime-stime) % 60;


	if(writeFlux)
	{
		strcpy(flxFiles, poreFileName);
		strcat(flxFiles, ".new");

		fout.open(flxFiles);
		for(k=0 ; k<nz  ; k++)
		for(j=0 ; j<ny  ; j++)
		for(i=ELayers ; i<nx-ELayers ; i++)
		{
			if(type[i][j][k] == '0') fout << "0\n";
			else                     fout << 'A'-type[i][j][k]-1 << "\n";
		}
		fout.close();
	}



	fout << "%=========================================================\n";
	fout << "% Porosity: " << porosity << ", Perm(mD): " << totPerm << "\n";
	fout << "% Calculation Time : " << min << " min " << sec << " sec\n";
	fout << "% Perm Calculation - normal termination (LBSPFS_IFP.1.1.0)\n";
	fout << "%=========================================================\n";


	fout.open(resFileName);

	fout << "%=========================================================\n";
	fout << "% LB Single-phase Flow Simulator Ver.IFP.1.1.0 - Mar. 2003\n";
	fout << "%                   Youngseuk Keehm - All rights reserved.\n";
	fout << "%---------------------------------------------------------\n";
	fout << "% Pore filename  : " << poreFileName << " (" << nxx << " " 
								  << ny << " " << nz << ")\n"; 
	fout << "% Porosity       : " << porosity << std::endl;
	fout << "% Grid spacing   : " << dx << "\n";
	fout << "% Extra Layers   : " << ELayers << "\n";
	fout << "% No. of Clusters: " << nC << "\n";
	fout << "%=========================================================\n";
	fout << "% Cal. Perm(mD)  : " << totPerm << "\n";
	fout << "% Cal. Time      : " << min << " min " << sec << " sec\n";
	fout << "%=========================================================\n";

	fout.close();



	return 0;
}

//////////////////////////////////////////////////////////////////////
// Function: calCurrentFlux()
//  Calculate current flux and add this contribution to total Q
//////////////////////////////////////////////////////////////////////
void Lattice::calCurrentFlux(int nCNodes, double* x)
{
	int i;
	double P1, P2;

	int n1, n2, n3, n4;

	n1 =  ELayers   *nInlet;
	n2 = (ELayers+1)*nInlet;
	n3 =  nCNodes - (ELayers+1)*nOutlet;
	n4 =  nCNodes -  ELayers   *nOutlet;

	// Calculate Inlet and Outlet Pressures
	P1 = P2 = 0.0;
	for(i=n1*4 ; i<n2*4 ; i+=4)
		P1 += x[i];

	for(i=n3*4 ; i<n4*4 ; i+=4)
		P2 += x[i];
	
	P1   = P1/nInlet;
	P2   = P2/nOutlet;
	dPdx = cs2*(P1 - P2)/(nx-2*ELayers-1);

	// Initialize total flux
	Q[0] = Q[1] = Q[2] = 0.0;


	int nxyz = (nx-2*ELayers)*ny*nz;

	for(i=n1*4 ; i<n4*4 ; i+=4){
		Q[0] += x[i+1];
		Q[1] += x[i+2];
		Q[2] += x[i+3];
	}
	perm = 1e9*(Q[0]/nxyz)*nu*dx*dx/dPdx;  

}

//////////////////////////////////////////////////////////////////////
// Function: solve()
//  Solve Lattice-Boltzmann equation by Bi-Conjugate Gradient
//  Stabilized method. Since this method does not form 
//  the Matrix A, we cannot use the preconditioner. If we want
//  the preconditioner, we need to form A and calculate 
//  pseudo-inverse or decomposition.
//
//  Ax = b <-->  (I-PTE)x = PTF 
//////////////////////////////////////////////////////////////////////
int Lattice::solve(int nCNodes, double *x, double *b)
{ 
	int    n = 4*nCNodes;
	double residual = 1.0, conv;
	int	   step, maxStep=1000;
	double normb, rho0, rho1;
	double beta, alpha,omega;
	double *r, *rp, *p, *v, *s, *t;
	double perm_old=0.0;
	int    convCount = 0;

	r  = new double[n];
	rp = new double[n];
	p  = new double[n];
	v  = new double[n];
	s  = new double[n];
	t  = new double[n];

	normb = VecNorm(n, b);

	MatVec(nCNodes, x, r);			// r  = Ax
	VecAXPY(n, -1.0, r, b, rp);		// rp = b - Ax
	VecCopy(n, rp, r);				// r  = rp

	step = 0;
	while( step<maxIter )
	{
		rho1 = VecDot(n, r, rp);

		if(rho1 == 0.0)						// Method fails
			return -1;		
		
		if(step == 0) {
			VecCopy(n, r, p);				// first step: p = r
		}
		else{
			beta = (rho1/rho0)*(alpha/omega);		
			VecAXPY(n,-omega, v, p, p);			// p = -omega*v + p
			VecAXPY(n, beta,  p, r, p);			// p = beta*p + r
		}

		MatVec(nCNodes, p, v);				// v = Ap
		alpha = rho1/VecDot(n, rp, v);		// alpha = rho1/(rp'*v)
		VecAXPY(n,-alpha, v, r, s);			// s = -alpha*v + r

		MatVec(nCNodes, s, t);				// t = As
		omega = VecDot(n,t,s)/VecDot(n,t,t);// omega = (t'*s)/(t'*t)
		VecAXPY(n, alpha, p, x, x);
		VecAXPY(n, omega, s, x, x);			// x = x + alpha*p + omega*s
		VecAXPY(n,-omega, t, s, r);			// r = -omega*t + s
		

		if((step+1)%dumpInterval == 0)
		{
			residual=VecNorm(n, r)/normb;
			calCurrentFlux(nCNodes, x);
			conv = fabs((perm-perm_old)/perm)/dumpInterval;
			perm_old = perm;

			if(conv<tol_perm) convCount++;
			else              convCount = 0;

			std::cout << step+1 << " " << perm << " " << residual << " " << conv << std::endl;
		
			//if(residual<tol_norm && conv<tol_perm)	break;
			//if(residual<1e-4 && convCount > 3) break;
			//if(residual<1e-6 && conv<1e-5) break;
			if(convCount > 3) break;
		}
		
		// check continuation condition
		if(omega == 0.0)
			return -2;

		// Update variables
		rho0 = rho1;
		step++;
	}


	// free allocated memory
	delete [] r;
	delete [] rp;
	delete [] p;
	delete [] v;
	delete [] s;
	delete [] t;

	return step;
}

//////////////////////////////////////////////////////////////////////
// General Vector operations
//	AXPY(ax+y=z), VecNorm(||v||), VecDot(v'*v), VecCopy(y=x), 
//  VecSet(y[i]=a)
//////////////////////////////////////////////////////////////////////
void Lattice::VecAXPY(int n, double coeff,double *x,double *y,double*z)
{
	int i;
	for(i=0; i<n; i++)
		z[i] = coeff*x[i] + y[i];
}

double Lattice::VecNorm(int n, double *y)
{
	return sqrt(VecDot(n, y, y));
}

double Lattice::VecDot(int n, double *x, double *y)
{
	int i;
	double sum=0.0;

	for(i=0; i<n; i++)
		sum += x[i]*y[i];

	return(sum);
}

void Lattice::VecCopy(int n, double *x, double *y)
{
	int i;
    for(i=0; i<n; i++) y[i] = x[i];
}

void Lattice::VecSet(int n, double v, double *x)
{
	int i;
    for(i=0; i<n; i++) x[i] = v;
}

//////////////////////////////////////////////////////////////////////
// Define Matrix-Vector muliplication
//	Ax operation which denotes
//  (I-PTE)x = Ix - P(T(Ex))
//  Instead of having explicit sparse matrix A, detailed operations 
//  will be performed to save memory and operation counts.
//
// nCNodes: number of nodes
// x      : input vector with length of (nCNodes*4)
// E      : expansion matrix (nCNodes*4 --> nCNodes*18)
// T      : transition matrix (nCNodes*18 --> nCNodes*18)
// P      : projection matrix (nCNodes*18 --> NCNodes*4)
// I      : Itentity matrix (NCNodes*4)
//////////////////////////////////////////////////////////////////////
void Lattice::MatVec(int nCNodes, double *x, double *y)
{
	int	   n, i;
	int    *up;
	double *xp, *yp, *tp;
	double tt[18];
	double rho1, rho2;
	double tmass=0.0;
	double mo11, mo12, mo21, mo22, mo31, mo32;

    // x'(tmp1) = Ex; map density and momenta to particle densities
    for (n=0; n<nCNodes ; n++)
    {
		xp = x + n*4;

		rho1 = xp[0]/12.0;
		rho2 = xp[0]/24.0;
		mo11 = xp[1]/6.0;
		mo12 = xp[1]/12.0;
		mo21 = xp[2]/6.0;
		mo22 = xp[2]/12.0;
		mo31 = xp[3]/6.0;
		mo32 = xp[3]/12.0;

		tp = tmp1 + n*np;

		tp[0]  = rho1 + mo11;
		tp[1]  = rho1 - mo11;
		tp[2]  = rho1 + mo21;
		tp[3]  = rho1 - mo21;
		tp[4]  = rho1 + mo31;
		tp[5]  = rho1 - mo31;

		tp[6]  = rho2 + mo12 + mo22;
		tp[7]  = rho2 - mo12 - mo22;
		tp[8]  = rho2 + mo12 - mo22;
		tp[9]  = rho2 - mo12 + mo22;

		tp[10] = rho2 + mo12 + mo32;
		tp[11] = rho2 - mo12 - mo32;
		tp[12] = rho2 + mo12 - mo32;
		tp[13] = rho2 - mo12 + mo32;

		tp[14] = rho2 + mo22 + mo32;
		tp[15] = rho2 - mo22 - mo32;
		tp[16] = rho2 + mo22 - mo32;
		tp[17] = rho2 - mo22 + mo32; 
	}

	tmp1[nCNodes*np] = tmp1[nCNodes*np+1] = 0.0;

    // x" = PTx'; particle densities to momenta

	for (n=0 ; n<nCNodes ; n++)
	{
		up = updates + n*np;
		for(i=0 ; i<np ; i++)
			tt[i] = tmp1[up[i]];

		xp = x + n*4;
		yp = y + n*4;

		// x_new =Ix - x"
		tmass += xp[0];
		yp[0] = xp[0] - (tt[0]+tt[1] +tt[2] +tt[3] +tt[4] +tt[5] +tt[6] +tt[7] +tt[8]
						+tt[9]+tt[10]+tt[11]+tt[12]+tt[13]+tt[14]+tt[15]+tt[16]+tt[17]);
		yp[1] = xp[1] - (tt[0]-tt[1] +tt[6] -tt[7] +tt[8] -tt[9] +tt[10]-tt[11]+tt[12]-tt[13]);
		yp[2] = xp[2] - (tt[2]-tt[3] +tt[6] -tt[7] -tt[8] +tt[9] +tt[14]-tt[15]+tt[16]-tt[17]);
		yp[3] = xp[3] - (tt[4]-tt[5] +tt[10]-tt[11]-tt[12]+tt[13]+tt[14]-tt[15]-tt[16]+tt[17]);
	}
}



//////////////////////////////////////////////////////////////////////
// Function: calRHS()
//	calculate RHS of Ax=b, where b = PB
// nCNodes: number of nodes
// B      : density from boundary with length of nCNodes*18
// P      : projection matrix (nCNodes*18 --> NCNodes*4)
//////////////////////////////////////////////////////////////////////
void Lattice::calRHS(int nCNodes, double* b)
{
	int    i, n;
	int    *up;
	double lF[18];
	double *bb;

	for(n=0 ; n<nCNodes*4 ; n++)			// Initialize b
		b[n] = 0.0;

	for(n=0	; n<nCNodes; n++)				// for all pore nodes in curr. cluster
	{
		up   = updates + n*np;

		for(i=0 ; i<np ; i++)
		{
			if(up[i] == nCNodes*np) 
				lF[i] = IForce[i];
			else if(up[i] == nCNodes*np+1) 
				lF[i] = OForce[i];
			else
				lF[i] = 0.0;
		}
		
		bb = b + n*4;

		bb[0] = lF[0] +lF[1] +lF[2] +lF[3] +lF[4] +lF[5] +lF[6] +lF[7] +lF[8] +lF[9]
			   +lF[10]+lF[11]+lF[12]+lF[13]+lF[14]+lF[15]+lF[16]+lF[17];
		bb[1] = lF[0] -lF[1] +lF[6] -lF[7] +lF[8] -lF[9] +lF[10]-lF[11]+lF[12]-lF[13];
		bb[2] = lF[2] -lF[3] +lF[6] -lF[7] -lF[8] +lF[9] +lF[14]-lF[15]+lF[16]-lF[17];
		bb[3] = lF[4] -lF[5] +lF[10]-lF[11]-lF[12]+lF[13]+lF[14]-lF[15]-lF[16]+lF[17];
	}
}


//////////////////////////////////////////////////////////////////////
// Function: clustering()
//	find clusters of the structure
//  Now pore nodes are assigned with capital letter.
//  For example, A: first cluster, B: second and so on.
//
//  Return    : number of clusters 
//  Dependency: require IntQueue class 
//////////////////////////////////////////////////////////////////////
int Lattice::clustering()
{
	int       i, j, k, ii, jj, kk;
	int       n, ind;
	int       nElem;
	bool      flag;
	char      currClustIndex = 'A';
	IntQueue  q(ny*nz*4);
	int       nClusters=0;

	while(true)
	{
		nElem = 0;

		for(k=0 ; k<nz ; k++)
		{
			for(j=0 ; j<ny ; j++)
			{
				if(type[0][j][k] == '0'){ 
					type[0][j][k] = currClustIndex;
					q.insert( nx*(ny*k + j) );
					nElem++;
					break;
				}
			}
			if(q.size()>0) break;
		}

		if(q.size() == 0) break;  // end of all connected pore

		flag = false;
		while( q.size() )
		{
			ind = q.remove();
			i   = ind % nx;
			j   = (int)(ind/nx) % ny;
			k   = (int)(ind/nx/ny);
			for(n=0 ; n<np ; n++)
			{
				ii = i+C[n][0];
				jj = j+C[n][1];
				kk = k+C[n][2];
				if( ii>=0 && ii<nx && jj>=0 && jj<ny && kk>=0 && kk<nz 
					&& type[ii][jj][kk] == '0' )
				{
					type[ii][jj][kk] = currClustIndex;
					q.insert(  nx*(ny*kk + jj) + ii );
					nElem++;
					if( ii == nx-1 ) flag = true;
				}
			}
		}
		
		if(flag)
		{
			nClustElem[nClusters] = nElem;
			clustIndex[nClusters] = currClustIndex;
			nClusters++;
		}
		currClustIndex++;
	}
	return nClusters;
}


//////////////////////////////////////////////////////////////////////
// Function: initNodes()
//	Setting up propagation indices (updates[])
//  The length of updates[] is nCNodes*np
//  For Intet, the target index is nCNodes*np
//  For Outlet, the target index is nCNodes*np+1
//
//  Also initialize the solution vector x with constant dPdx
//////////////////////////////////////////////////////////////////////
void Lattice::initNodes(char currIndex, int nCNodes, double *x)
{
	/* temporary variables */
	int    i, j, k, mm, ii, jj, kk, n, nn, n2;

	/* Allocate temporary index */
	int ***Index = new int**[nx];
	for(i=0 ; i<nx ; i++){
		Index[i] = new int*[ny];
		for(j=0 ; j<ny ; j++){
			Index[i][j] = new int[nz];
		}
	}
	
	// Setting up Index for pore nodes
	//  by simply assigning continuous integer.
	n = nInlet = nOutlet =0;
	for(i=0 ; i<nx ; i++)
	for(j=0 ; j<ny ; j++)
	for(k=0 ; k<nz ; k++)
	{ 
		if(type[i][j][k] == currIndex){
			Index[i][j][k] = n++;

			if(i==0)    nInlet++;
			if(i==nx-1) nOutlet++;
		}
	}
	
	// Checking correctness of index
	if( nCNodes != n ) {
		std::cout << "Error! in setting the number of Index[][][]\n";
		exit(0);
	}

	// Setting up propagation mapping indices.

	double dMass = DP/(double)(nx+1);

	n = 0;
	n2= 0;
	for(i=0 ; i<nx ; i++)
	for(j=0 ; j<ny ; j++)
	for(k=0 ; k<nz ; k++)
	{
    	if(type[i][j][k] == currIndex)
    	{
			x[n2]   = 1.0 + dMass*(double)(nx-i);
			x[n2+1] = 0.0;
			x[n2+2] = 0.0;
			x[n2+3] = 0.0;
			n2 += 4;

			for(mm = 0 ; mm<np ; mm++)
			{
    			ii = i - C[mm][0];
				jj = j - C[mm][1];
				kk = k - C[mm][2];

				if(ii<0)
				{
					//if(jj<0 || jj>=ny || kk<0 || kk>=nz || type[0][jj][kk] != currIndex)
						updates[n] = nCNodes*np;
					//else{
					//	if( mm%2 == 0 ) nn=mm+1;
					//	else            nn=mm-1;
					//	updates[n] = nn + np*Index[i][j][k];
					//}
				}
				else if(ii>=nx)
				{
					//if(jj<0 || jj>=ny || kk<0 || kk>=nz || type[nx-1][jj][kk] != currIndex)
						updates[n] = nCNodes*np + 1;
					//else{
					//	if( mm%2 == 0 ) nn=mm+1;
					//	else            nn=mm-1;
					//	updates[n] = nn + np*Index[i][j][k];
					//}
				}
				else if(jj<0 || jj>=ny || kk<0 || kk>=nz || 
					    type[ii][jj][kk] != currIndex)
				{ 	
					if( mm%2 == 0 ) nn=mm+1;
					else            nn=mm-1;
					updates[n] = nn + np*Index[i][j][k];
				}
				else
					updates[n] = mm + np*Index[ii][jj][kk];
		
				n++;	
			}
		}
	}

    // Checking correctness of assigning updates[]
	if( nCNodes*np != n ) {
		std::cout << "nNodes*np = " << nNodes*np << ", number of Nodes=" << n << std::endl;
		std::cout << "Error! in setting up updates[]\n";
		exit(0);
	}

	// Free up the Index array and pore geometry (Index & type)
	for(i=0 ; i<nx ; i++)
	{
	  for(j=0 ; j<ny ; j++){
		   delete [] Index[i][j];
	  }
	  delete [] Index[i];
	}
	delete [] Index;

}
		
// end of file