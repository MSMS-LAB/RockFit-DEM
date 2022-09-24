//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <cuda.h>
//#include <curand.h>
//#include <math_functions.h>
#include "md.h"
//#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
//#include <thrust/count.h>
//#include <thrust/device_allocator.h>
//#include <thrust/device_ptr.h>
#include <time.h>
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "pcuda_helper.h"

//#include <cudpp.h>
//#include <cudpp_plan.h>

//__constant__ char IM[2 * IMatrixSize];

void h_CheckInteractions(particle_data& P, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL)
{
	uint_fast32_t i, j, k, kij;
	uint3 ic, jc;
	float rr;
	float3 r, ics;
	
	for (i = 0; i < P.N; ++i)
	{
		for (j = 0; j < P.N; ++j)
		{
			if (i == j) continue;
			r.x = P.h_R[j] - P.h_R[i];
			r.y = P.h_R[j + P.N] - P.h_R[i + P.N];
			r.z = P.h_R[j + 2 * P.N] - P.h_R[i + 2 * P.N];
			rr = r.x * r.x + r.y * r.y + r.z * r.z;
			if (rr<IL.aacut)
			{
				kij = UINT_FAST32_MAX;
				for (k = i * IL.IonP; k < (i + 1) * IL.IonP; ++k)
				{
					if (IL.h_IL[k] == j)
						kij = k;
				}
				if (kij == UINT_FAST32_MAX)
				{
					ic.x = floor(P.h_R[i] * C._1d_a);
					ic.y = floor(P.h_R[i + P.N] * C._1d_a);
					ic.z = floor(P.h_R[i + 2 * P.N] * C._1d_a);
					jc.x = floor(P.h_R[j] * C._1d_a);
					jc.y = floor(P.h_R[j + P.N] * C._1d_a);
					jc.z = floor(P.h_R[j + 2 * P.N] * C._1d_a);
					ics.x = P.h_R[i] - (ic.x + 0.5) * C.a;
					ics.y = P.h_R[i + P.N] - (ic.y + 0.5) * C.a;
					ics.z = P.h_R[i + 2 * P.N] - (ic.z + 0.5) * C.a;
					std::cerr << "NOT find " << i << " " << j << " " << sqrt(rr) << " " << sqrt(IL.aacut) << "\n";
					std::cerr << "NOT find " << P.h_R[i] << " " << P.h_R[i + P.N] << " " << P.h_R[i + 2 * P.N]
						<< " | " << P.h_R[j] << " " << P.h_R[j + P.N] << " " << P.h_R[j + 2 * P.N] << "\n";
					std::cerr << "NOT find " << ic.x << " " << ic.y << " " << ic.z << " " << ic.x + C.Nr.x * ic.y + C.Nr.x * C.Nr.y * ic.z
						<< " | " << jc.x << " " << jc.y << " " << jc.z << " " << jc.x + C.Nr.x * jc.y + C.Nr.x * C.Nr.y * jc.z
						<< " | " << ics.x << " " << ics.y << " " << ics.z << " | " << 1.0 / C._1d_a << "\n";
					for (int ii = 0; ii < P.N; ++ii)
						std::cerr << ii << " " << C.h_CIs[ii] << " " << C.h_CIs[ii + P.N] << " " << C.h_CI[ii] << "\n";
					for (int ii = 0; ii < C.N; ++ii)
						if(C.h_pnC[ii]>0)
						std::cerr << ii << " " << C.h_pnC[ii] << " " << C.h_pnC[ii + C.N] << "\n";
					std::cin.get();
				}
			}
			if (rr < 1e-10)
				std::cerr << "CLOSE " << i << " " << j << " " << sqrt(rr) << " " << sqrt(IL.aacut) << "\n";
		}
	}
	for (i = 0; i < P.N; ++i)
	{
		for (k = i * IL.IonP; k < (i + 1) * IL.IonP; ++k)
		{
			j = IL.h_IL[k];
			if (j == UINT_FAST32_MAX)continue;
			r.x = P.h_R[j] - P.h_R[i];
			r.y = P.h_R[j + P.N] - P.h_R[i + P.N];
			r.z = P.h_R[j + 2 * P.N] - P.h_R[i + 2 * P.N];
			rr = r.x * r.x + r.y * r.y + r.z * r.z;
			if (rr > IL.aacut)
			{
				std::cerr << "Too FAR " << i << " " << j << " " << sqrt(rr) << " " << sqrt(IL.aacut) << "\n";
			}
		}		
	}
}

void CheckDATA(particle_data& P, potential_data &Po, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL)
{	
	//thrust::device_ptr<float> dtp_1d_IL(IL.d_1d_iL);
	//thrust::device_ptr<float> max_pIL = thrust::min_element(dtp_1d_IL, dtp_1d_IL + P.N);
	//unsigned int pIL = max_pIL - dtp_1d_IL;
	//float vIL = *max_pIL;	
	//thrust::device_ptr<float> dtp_1d_OT(IL.d_Oijt);
	//thrust::device_ptr<float> max_pOT = thrust::min_element(dtp_1d_OT, dtp_1d_OT + P.N);
	//unsigned int pIL = max_pOT - dtp_1d_OT;
	//float vIL = *max_pOT;
	uint_fast32_t i, ii, jj;
	double3 rii, rjj;
	//HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_Oijt, IL.d_Oijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(IL.h_rij, IL.d_rij, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(P.h_W, P.d_W, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
	double MaxMin[10] = { 0, 0, 0, 0, 0, 1e+20,0,0,0,0 }, val[5];
	for (i = 0; i < IL.N; ++i)
	{
		val[0] = IL.h_1d_iL[i];
		val[1] = (IL.h_Oijt[i].x* IL.h_Oijt[i].x+ IL.h_Oijt[i].y* IL.h_Oijt[i].y+ IL.h_Oijt[i].z* IL.h_Oijt[i].z);
		val[2] = IL.h_rij[i].x * IL.h_rij[i].x + IL.h_rij[i].y * IL.h_rij[i].y + IL.h_rij[i].z * IL.h_rij[i].z;
		if (MaxMin[1] < val[0])
		{
			MaxMin[0] = i;
			MaxMin[1] = val[0];
		}
		if (MaxMin[3] < val[1])
		{
			MaxMin[2] = i;
			MaxMin[3] = val[1];
		}
		if (MaxMin[5] > val[2] && IL.h_IL[i] != UINT_FAST32_MAX && IL.h_ILtype[i] == 1)
		{
			MaxMin[4] = i;
			MaxMin[5] = val[2];
		}
	}
	for (i = 0; i < P.N; ++i)
	{
		val[3] = P.h_V[i] * P.h_V[i] + P.h_V[i + P.N] * P.h_V[i + P.N] + P.h_V[i + 2 * P.N] * P.h_V[i + 2 * P.N];
		val[4] = P.h_W[i] * P.h_W[i] + P.h_W[i + P.N] * P.h_W[i + P.N] + P.h_W[i + 2 * P.N] * P.h_W[i + 2 * P.N];
		if (MaxMin[7] < val[3])
		{
			MaxMin[6] = i;
			MaxMin[7] = val[3];
		}
		if (MaxMin[9] < val[4])
		{
			MaxMin[8] = i;
			MaxMin[9] = val[4];
		}
	}

	ii = int(MaxMin[4] / IL.IonP);
	jj = IL.h_IL[int(MaxMin[4])];
	rii.x = P.h_R[ii]; rii.y = P.h_R[ii + P.N]; rii.z = P.h_R[ii + 2 * P.N];
	rjj.x = P.h_R[jj]; rjj.y = P.h_R[jj + P.N]; rjj.z = P.h_R[jj + 2 * P.N];
	std::cerr << "Min IL " << MaxMin[0] << " " << 1.0 / MaxMin[1] << " " << 1.0 / MaxMin[1] - 2.0 * Po.p_r
		<< " | Max O " << MaxMin[2] << " " << sqrt(MaxMin[3])
		<< " | Min r " << MaxMin[4] << " " << ii << " " << jj << " " << int(IL.h_ILtype[int(MaxMin[4])]) << " " << sqrt(MaxMin[5])
		<< " | " << rii.x << " " << rii.y << " " << rii.z << " " << rjj.x << " " << rjj.y << " " << rjj.z << "\n";
	std::cerr << "Max PV " << MaxMin[6] << " " << sqrt(MaxMin[7]) << " Max PW " << MaxMin[8] << " " << sqrt(MaxMin[9]) << "\n";


}
