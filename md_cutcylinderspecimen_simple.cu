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

__global__ void d_CutCylinderSpecimen_simple(float* __restrict__ R, float* __restrict__ V, const uint_fast32_t N, const float3 center, const float RR0, const float H0, const float3 hidenpoint)
{
	uint_fast32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (idx == 0)printf("In %e %e %e %e |%e %e %e %e | %e %e %e %e | %f %f\n", FV[idx], FV[idx + n], FU[idx], FU[idx + n], VV[idx], VV[idx + n], VU[idx], VU[idx + n], V[idx], V[idx + n], U[idx], U[idx + n], P_dtm, P_dt);
	//if(blockIdx.x > 3)printf("Inc %u %u %u %u %u\n", idx, n, threadIdx.x, blockIdx.x, blockDim.x);
	float3 dr;
	if (idx < N)
	{
		dr.x = R[idx] - center.x;
		dr.y = R[idx + N] - center.y;
		dr.z = R[idx + 2 * N] - center.z;	
		float rr = __fmul_rn(dr.x, dr.x) + __fmul_rn(dr.y, dr.y);
		if (rr > RR0)
		{
			R[idx] = hidenpoint.x;
			R[idx + N] = hidenpoint.y;
			R[idx + 2 * N] = hidenpoint.z;
			V[idx] = 0;
			V[idx + N] = 0;
			V[idx + 2 * N] = 0;
		} 
		if (fabsf(dr.z) > H0)
		{
			R[idx] = hidenpoint.x;
			R[idx + N] = hidenpoint.y;
			R[idx + 2 * N] = hidenpoint.z;
			V[idx] = 0;
			V[idx + N] = 0;
			V[idx + 2 * N] = 0;
			//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		} 		
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N]);
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, R[idx], R[idx + N], R[idx + 2 * N], s.x, s.y, s.z);
		//printf("C %i | %f %f %f | %f %f %f | %f %f\n", idx, dr.x, dr.y, dr.z, dm, sm, hm, coefr, coefh);		
	}	
}