#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <iostream>

__device__ void addlink(const float* __restrict__ R, const uint_fast32_t N, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t idx, const uint_fast32_t nindx, const uint_fast32_t jindx, uint_fast32_t* __restrict__ IL, uint_fast32_t IonP, float aacut, uint_fast32_t &nkndx)
{
	uint_fast32_t i, jdx;
	float rr;
	float3 dr;
	for (i = 0; i < nindx; ++i)
	{
		jdx = CIs[jindx + i + N];
		dr.x = R[jdx] - R[idx];
		dr.y = R[jdx + N] - R[idx + N];
		dr.z = R[jdx + 2 * N] - R[idx + 2 * N];
		rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y) + __fmul_rn(dr.z, dr.z);
		//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
		if (rr < aacut && jdx != idx)
		{
			//printf("I %i %i %u | %f\n", idx, jdx, jindx, sqrt(rr));
			IL[idx * IonP + nkndx] = jdx;
			++nkndx;
		}
	}	
}

__global__ void d_ConstructInteractionList(const float* __restrict__ R, uint_fast32_t N, const uint_fast32_t* __restrict__ CI, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t* __restrict__ pnC, uint_fast32_t* __restrict__ IL, uint_fast32_t IonP, float a, float _1d_a, float aacut, uint3 cN, uint_fast32_t CN)
{
	// set thread ID
	//uint_fast32_t tid = threadIdx.x;
	// global index
	uint_fast32_t idx = blockIdx.x * blockDim.x + threadIdx.x, iindx, nindx, jindx, jjindx, nkndx;
	uint3 indx;
	int3 indxx;
	//float rr;
	//float3 dr;
	//if(idx==0)printf("T %i %i | %f %f %f | %f %f %f \n", N, idx, A.x, A.y, A.z, L.x, L.y, L.z);
	// boundary check
	//if(idx==0)
	while (idx < N)
	{
		indx.x = floorf(__fmul_rn(R[idx], _1d_a));
		indx.y = floorf(__fmul_rn(R[idx + N], _1d_a));
		indx.z = floorf(__fmul_rn(R[idx + 2 * N], _1d_a));
		iindx = indx.x + indx.y * cN.x + indx.z * cN.y * cN.x;
		indxx.x = copysignf(1.0f, (R[idx] - __fmul_rn(indx.x + 0.5f, a)));
		indxx.y = copysignf(1.0f, (R[idx + N] - __fmul_rn(indx.y + 0.5f, a)));
		indxx.z = copysignf(1.0f, (R[idx + 2 * N] - __fmul_rn(indx.z + 0.5f, a)));
		//printf("CILf %i | %f %f %f %f | %u %u %u | %i %i %i\n", idx, R[idx], R[idx + N], R[idx + 2 * N], a, indx.x, indx.y, indx.z, indxx.x, indxx.y, indxx.z);
			
		nkndx = 0;
		jjindx = iindx;
		jindx = pnC[jjindx];
		nindx = pnC[jjindx + CN];
		//printf("pnC %i %u %u %u\n", idx, jindx, nindx, CIs[jindx + 0 + N]);
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		jjindx = iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x;
		jindx = pnC[jjindx];	nindx = pnC[jjindx + CN];
		addlink(R, N, CIs, idx, nindx, jindx, IL, IonP, aacut, nkndx);
		//printf("T1 %u | %u %u %u %u %u %u %u %u | %u %u %u\n", idx, iindx, iindx + indxx.x, iindx + indxx.y * cN.x, iindx + indxx.z * cN.y * cN.x,
		//	iindx + indxx.x + indxx.y * cN.x, iindx + indxx.x + indxx.z * cN.y * cN.x, iindx + indxx.y * cN.x + indxx.z * cN.y * cN.x,
		//	iindx + indxx.x + indxx.y * cN.x + indxx.z * cN.y * cN.x, pnC[jjindx], pnC[iindx + CN], nkndx);
#ifdef pre_debugtest
		if (nkndx > IonP)
			printf("Error! d_ConstructInteractionList %i %u %u\n", idx, nkndx, IonP);
#endif // pre_debugtest

		
		//printf("T %i %u\n", idx, nkndx);
		idx += blockDim.x * gridDim.x;
	}
}



