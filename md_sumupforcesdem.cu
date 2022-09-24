#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <math_functions.h>
#include "md.h"
#include "pcuda_helper.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <time.h>

//#include <cudpp.h>
//#include <cudpp_plan.h>

//__constant__ char IM[2 * IMatrixSize];
__global__ void d_SumUpForcesDEM(const uint_fast32_t* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const	uint_fast32_t N, const uint_fast32_t IonP)
{
	//float rr, fm, _1d_r;
	//float3 dr, fs;
	uint_fast32_t idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, kdx;

	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{
		//uint_fast8_t type;
		//kdx = idx * IonP;
		//jdx = IL[kdx];
		//type = ILtype[kdx];
		//fs.x = 0; fs.y = 0; fs.z = 0; ii = 0;
		float3 fsum = { 0,0,0 }, msum = { 0,0,0 };
		//for (jdx < N && kdx < (idx + 1) * IonP)
		for (kdx = idx * IonP; kdx < (idx + 1) * IonP; ++kdx)
		{			
			fsum.x += Fijn[kdx].x + Fijt[kdx].x;
			fsum.y += Fijn[kdx].y + Fijt[kdx].y;
			fsum.z += Fijn[kdx].z + Fijt[kdx].z;
			msum.x += Mijn[kdx].x + Mijt[kdx].x + Mijadd[kdx].x;
			msum.y += Mijn[kdx].y + Mijt[kdx].y + Mijadd[kdx].y;
			msum.z += Mijn[kdx].z + Mijt[kdx].z + Mijadd[kdx].z;
			//++kdx;
			//jdx = IL[kdx];
			//type = ILtype[kdx];
		}
		//printf("FI %u %e %e %e\n", idx, fsum.x, fsum.y, fsum.z);

		F[idx] = fsum.x;
		F[idx + N] = fsum.y;
		F[idx + 2 * N] = fsum.z;// -10000 * 0.00170824f;
		M[idx] = msum.x;
		M[idx + N] = msum.y;
		M[idx + 2 * N] = msum.z;
		idx += blockDim.x * gridDim.x;
	}
}

__global__ void d_SumUpForcesDEMViscos(const uint_fast32_t* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const float* __restrict__ V, const float* __restrict__ W, const	uint_fast32_t N, const uint_fast32_t IonP, const float nuV, const float nuW)
{
	//float rr, fm, _1d_r;
	//float3 dr, fs;
	uint_fast32_t idx = blockIdx.x * blockDim.x + threadIdx.x, jdx, kdx;

	//printf("IM %i %i %i %i %i %i %i\n", IM[0], IM[1], IM[2], IM[3], IM[4], IM[5], IM[6]);
	//if(idx==0)
	//if(idx==0)printf("AAAAA\n");
	while (idx < N)
	{
		//uint_fast8_t type;
		kdx = idx * IonP;
		jdx = IL[kdx];
		//type = ILtype[kdx];
		//fs.x = 0; fs.y = 0; fs.z = 0; ii = 0;
		float3 fsum = { 0,0,0 }, msum = { 0,0,0 };
		//for (jdx < N && kdx < (idx + 1) * IonP)
		for (kdx = idx * IonP; kdx < (idx + 1) * IonP; ++kdx)
		{
			fsum.x += Fijn[kdx].x + Fijt[kdx].x;
			fsum.y += Fijn[kdx].y + Fijt[kdx].y;
			fsum.z += Fijn[kdx].z + Fijt[kdx].z;
			msum.x += Mijn[kdx].x + Mijt[kdx].x + Mijadd[kdx].x;
			msum.y += Mijn[kdx].y + Mijt[kdx].y + Mijadd[kdx].y;
			msum.z += Mijn[kdx].z + Mijt[kdx].z + Mijadd[kdx].z;
			//++kdx;
			//jdx = IL[kdx];
			//type = ILtype[kdx];
		}
		//fsum.x -= V[idx] * nuV;
		//fsum.y -= V[idx + N] * nuV;
		//fsum.z -= V[idx + 2 * N] * nuV;
		//msum.x -= W[idx] * nuW;
		//msum.y -= W[idx + N] * nuW;
		//msum.z -= W[idx + 2 * N] * nuW;
		//printf("FI %u %e %e %e\n", idx, fsum.x, fsum.y, fsum.z);

		F[idx] = fsum.x;
		F[idx + N] = fsum.y;
		F[idx + 2 * N] = fsum.z;// -10 * 0.00170824f;
		M[idx] = msum.x;
		M[idx + N] = msum.y;
		M[idx + 2 * N] = msum.z;
		idx += blockDim.x * gridDim.x;
	}
}