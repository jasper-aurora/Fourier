// Steady wave program - C++ version

// cc -o Fourier Dpythag.cpp Dsvbksb.cpp Dsvdcmp.cpp Fourier.cpp Inout.cpp Solve.cpp Subroutines.cpp Util.cpp // JP

#include <math.h>
#include <stdio.h>
// #include <process.h> // JP
#include <string.h>
// #include <conio.h> // JP
#include <stdlib.h>
#define	ANSI
#include "Allocation.h"

void
	init(void), Solve(void), Title_block(FILE*),	Output(void);

#define Main
#define	Int		int
#define	Double	double
#include "Headers.h"
#define	Diagnostic
#ifdef Diagnostic
char Diagname[30], Theory[10], theory=0, Um[2];
#endif

int main(void)
{
int	i, j, iter, m;
int	Read_data(void);

double	Newton(int), dhe, dho, error;
void 	Powell(double *, double **, int, double, int *, double *,double (*)(double *));

Input1 = fopen("Data.dat","r");

strcpy(Convergence_file,"Convergence.dat");
strcpy(Points_file,"Points.dat");
monitor = stdout;
strcpy(Theory,"Fourier");
strcpy(Diagname,"Catalogue.res");

Read_data();
num=2*n+10;
dhe=Height/nstep;
dho=MaxH/nstep;

/*CC = dmatrix(1,num,1,num);
for ( j=1; j <=num ; ++j)
	{
	for ( i=1; i <=num ; ++i)
		CC[j][i] = 0.;
	CC[j][j] = 1.;
	}
*/
Y = dvector(0,num);
z = dvector(1,num);
rhs1 = dvector(1,num);
rhs2 = dvector(1,num);
coeff = dvector(0, n);
cosa = dvector(0,2*n);
sina = dvector(0,2*n);
sol = dmatrix(0,num,1,2);
B = dvector(1, n);
Tanh = dvector(1,n);

//	Commence stepping through steps in wave height

for ( ns = 1 ; ns <= nstep ; ns++ )
	{
	height=ns*dhe;
	Hoverd=ns*dho;
	fprintf(monitor,"\n\nHeight step %2d of %2d\n", ns, nstep);

//	Calculate initial linear solution

	if(ns <= 1) init();

//	Or, extrapolate for next wave height, if necessary

	else
		for ( i=1 ; i <= num ; i++ )
			z[i]=2.*sol[i][2]-sol[i][1];

//	Commence iterative solution

	for (iter=1 ; iter <= number ; iter++ )
		{
		fprintf(monitor,"\nIteration%3d:", iter);

//	Calculate right sides of equations and differentiate numerically
//	to obtain Jacobian matrix, then solve matrix equation

		error = Newton(iter);

//	Convergence criterion satisfied?

		fprintf(stdout," Mean of corrections to free surface: %8.1e", error);
		if(ns == nstep)	criter = 1.e-10 ;
		else			criter = crit;
		if((error < criter * fabs(z[1]))  && iter > 1 ) break;
		if(iter == number)
			{
			fprintf(stdout,"\nNote that the program still had not converged to the degree specified\n");
			}

//	Operations for extrapolations if more than one height step used

		if(ns == 1)
			for ( i=1 ; i<=num ; i++ )
				sol[i][2] = z[i];
		else
			for ( i=1 ; i<=num ; i++ )
				{
				sol[i][1] = sol[i][2];
				sol[i][2] = z[i];
				}
		} // End of iteration solution loop

//	Fourier coefficients (for surface elevation by slow Fourier transform)

for ( Y[0] = 0., j = 1 ; j <= n ; j++ )
	{
   B[j]=z[j+n+10];
	sum = 0.5*(z[10]+z[n+10]*pow(-1.,(double)j));
	for ( m = 1 ; m <= n-1 ; m++ )
		sum += z[10+m]*cosa[(m*j)%(n+n)];
	Y[j] = 2. * sum / n;
	}
} // End stepping through wave heights

// Print  results

Solution=fopen("Solution.res","w");
Solution2=fopen("Solution-Flat.res","w");
Elevation = fopen("Surface.res","w");
Flowfield = fopen("Flowfield.res","w");

Output();

// _fcloseall(); //26.01.2019 // JP
fflush(NULL); // JP

// printf("\nTouch key to continue\n\n"); getch(); // JP

printf("\nFinished\n");
} // End main program

