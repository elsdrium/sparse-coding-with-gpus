#include <iostream>
#include "l1ls_coord_descent.hh"
#include "coreutils.hh"
#include "math.h"

float norm1(const float* x, int n) {
	float x1norm = 0;
	for(int i=0;i<n; ++i) {
		x1norm += fabs(x[i]);
	}
	return x1norm;
}

//float fobj(const float* x, const Matrix& A, const float* y, float gamma) {
//	DEBUGIFY(std::cerr << "fobj[begin]"<<std::endl);
//	// float f= 0.5 * norm(y-A*x)^2;
//	float sum_sqr_y_minus_Ax = 0;
//	float x1norm = 0;
//	for(int i=0; i< num_rows(A); i++) {
//		float row_Ax = 0;
//		for(int j=0; j<num_cols(A);j++) {
//			row_Ax += get_val(A,i,j)*x[j];
//			if(i==0) {
//				x1norm += fabs(x[j]); // this required later
//			}
//		}
//		float row_y_minus_Ax = y[i]-row_Ax;
//		row_y_minus_Ax *= row_y_minus_Ax;
//		sum_sqr_y_minus_Ax += row_y_minus_Ax;
//	}
//	float f = sqrt(sum_sqr_y_minus_Ax);
//	f += gamma*x1norm;
//	// f= f+ gamma*norm(x,1);
//	DEBUGIFY(std::cerr << "fobj[end]"<<std::endl);
//	return f;
//}

void l1ls_coord_descent_sub(float* xout, 
		float gamma, 
		const Matrix& A,
		const float* y,
	    const float* xinit,
	    const Matrix& AtA) {
//	DEBUGIFY(std::cerr << "l1ls_coord_descent_sub[begin]"<<std::endl);
	const int num_alphas = 5;
	float alphas[num_alphas] = { 1, 3e-1, 1e-1, 3e-2, 1e-2 };
	float tol = 1e-6;
	int n = num_cols(A);
	int k = num_rows(A);
	
	float* x = (float*) malloc(sizeof(float)*n);
//	DEBUGIFY(std::cerr<<"x\n");
	for(int i=0;i<n;++i) {
		x[i] = 0;//xinit[i];
//		DEBUGIFY(std::cerr<<x[i]<<" ");
	}
	float* ytA = (float*) malloc(sizeof(float)*n);
	for(int i=0;i<n;++i) {
		float temp=0;
		for(int j=0;j<k;++j) {
			temp+=y[j]*get_val(A,j,i);
		}
		ytA[i]=temp;
//		printf("\nyta: %g, [i:%d]",ytA[i],i);
	}
	float* xstar = (float*) malloc(sizeof(float)*n);
	float* y_minus_Ax_t_A = (float*)malloc(sizeof(float)*n);
	float* d = (float*) malloc(sizeof(float)*n);
	float* xn = (float*) malloc(sizeof(float)*n);
	float* dtAtA = (float*)malloc(sizeof(float)*n);
	for(int iter=0;iter<700;++iter) {
//		DEBUGIFY(std::cerr<< "\nIter: "<<iter<<std::endl);
		// compute y_minus_Ax_t_A = ytA - x'*AtA
//		DEBUGIFY(std::cerr<<"\nComputing y_minus_Ax_t_A\n");
		for(int i=0;i<n;++i) {
			float temp = 0;
			for(int j=0;j<n;++j)
				temp += x[j]*get_val(AtA,j,i);
			y_minus_Ax_t_A[i] = ytA[i]-temp;
//			DEBUGIFY(printf("\ny_minus_ax_t_a: %g",y_minus_Ax_t_A[i]));
//			DEBUGIFY(printf("\nata[%d,%d]: %g",i,i,get_val(AtA,i,i)));
		}
		//		DEBUGIFY(std::cerr<<"Computing xstar\n");
		for(int j=0; j<n; ++j) {
			float Pj = 0.5*get_val(AtA,j,j);
			float Qj = y_minus_Ax_t_A[j] + get_val(AtA,j,j)*x[j];
			float Dj = -Qj;
			if(fabs(Dj)< gamma)
				xstar[j] = 0;
			else if(Dj > gamma)
				xstar[j] = (Qj + gamma)/(2.0*Pj);
			else
				xstar[j] = (Qj - gamma)/(2.0*Pj);
//			DEBUGIFY(printf("\nAtA[%d,%d]: %g",j,j,get_val(AtA,j,j)));
//			DEBUGIFY(printf("\nDi: %g, Pi: %g, Qi: %g",Dj,Pj,Qj));
			//			DEBUGIFY(std::cerr<<xstar[j]<<" ");
		}
		for(int i=0;i<n;++i) {
			d[i] = xstar[i]-x[i];
//			DEBUGIFY(printf("\nxstar[%d]: %g",i,xstar[i]));
//			DEBUGIFY(printf("\nx[%d]: %g",i,x[i]));
//			DEBUGIFY(printf("\nd[%d]: %g",i,d[i]));
		}
		float a = 0; // compute a = 0.5*d'*AtA*d
		//		DEBUGIFY(std::cerr<<"Computing dtAtA\n");
		for(int i=0;i<n; ++i) {
			float temp=0.0;
			for(int j=0;j<n;++j)
				temp += get_val(AtA,j,i)*d[j];
			dtAtA[i] = temp;
//			DEBUGIFY(printf("\ndtAtA: %g",dtAtA[i]));
		}
		for(int i=0;i<n;++i) {
			a += 0.5*dtAtA[i]*d[i];
		}
		//		DEBUGIFY(std::cerr<<"a: "<<a<<std::endl);

		float b = 0;
		// compute b = - y_minus_Ax_t_A*d
		for(int i=0;i<n; ++i) {
			b = b - y_minus_Ax_t_A[i]*d[i];
		}
//		DEBUGIFY(printf("\naf: %g",a));
//		DEBUGIFY(printf("\nbf: %g",b));
		//		DEBUGIFY(std::cerr<<"\nb: "<<b <<std::endl);
		float minhx = gamma*norm1(x,n);
		int imina = -1;
		bool found = false;
		int ia=0;
		while(ia<num_alphas && !found) {
			float alpha = alphas[ia];
			for(int i=0;i<n;++i) {
				xn[i] = x[i]+alpha*d[i];
			}
			float hx = a*alpha*alpha + b*alpha + gamma*norm1(xn,n);
			if(hx < minhx*(1-tol)) {
				imina = ia;
				minhx = hx;
				found = true;
			}
			ia++;
		}
//		for(int ia=0;ia<num_alphas;++ia) {
//			float alpha = alphas[ia];
//			for(int i=0;i<n;++i) {
//				xn[i] = x[i]+alpha*d[i];
//			}
//			float hx = a*alpha*alpha + b*alpha + gamma*norm1(xn,n);
//			if(hx < minhx*(1-tol)) {
//				imina = ia;
//				minhx = hx;
//				break; // might not b best for cuda
//			}
//		}
		if(imina==-1) {
			printf("breaking becahse imina==-1");
//			DEBUGIFY(std::cerr<<"breaking because imina==0\n");
			break;
		}
//		DEBUGIFY(printf("\nimina: %d",imina));
		for(int i=0;i<n;++i) {
			x[i] = x[i] + alphas[imina]*d[i];
		}
	}
	free(dtAtA);
	free(xn);
	free(d);
	free(y_minus_Ax_t_A);

//	DEBUGIFY(std::cerr<<"Updating xout\n");
	for(int i=0;i<n;++i) {
//		DEBUGIFY(printf("\nx[%d]: %g",i,x[i]));
//		DEBUGIFY(std::cerr<< x[i] << " from " << xout[i] << "\n");
		xout[i] = x[i];
	}
	free(x);
	free(xstar);
	free(ytA);
//	DEBUGIFY(std::cerr << "l1ls_coord_descent_sub[end]"<<std::endl);
}

void l1ls_coord_descent (Matrix& Xout, /* : output, size: n, m */ 
		float gamma, /* : input */
		const Matrix& A, /* : input, size: k, n */
		const Matrix& Y, /* : input, size: k, m */
		const Matrix& Xinit) { /*: input, size: n, m */
//	DEBUGIFY(std::cerr << "l1ls_coord_descent[begin]"<<std::endl);
	// AtA...
	Matrix AtA;
	init(AtA,num_cols(A),num_cols(A));
	init(Xout, num_rows(Xinit),num_cols(Xinit),false);
//	printf("Outputing AtA:\n");
	for(int i=0; i<num_cols(A); ++i) {
//		printf("\nRow %d: ",i);		
		for(int j=0; j<num_cols(A); ++j) {
			float temp=0.0;
			for(int l=0; l<num_rows(A); ++l) {
				temp+=get_val(A,l,i)*get_val(A,l,j);
			}
			set_val(AtA,i,j,temp);
//			printf("%g ",temp);
//			std::cerr<<get_val(AtA,i,j)<<" ";
		}
	}
	printf("\n");
	//... AtA
	for(int i=0; i<num_cols(Y); i++) {
//		DEBUGIFY(printf("im: %d\n",i));
//		if(i%10==0)
//			std::cerr<<".";
//		DEBUGIFY(std::cerr << "l1ls_coord_descent: " << i << "/" << num_cols(Y) << std::endl);
		const float* xinit = get_col(Xinit,i);
		float* xout = get_col(Xout,i);
		l1ls_coord_descent_sub(xout,gamma,A,get_col(Y,i),xinit,AtA);
	}
	freeup(AtA);
//	DEBUGIFY(std::cerr << "l1ls_coord_descent[end]"<<std::endl);
}
