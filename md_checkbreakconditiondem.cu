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

__global__ void d_CheckBreakConditionDEM(const uint_fast32_t* __restrict__ IL, const float* __restrict__ _1d_iL, 
	const float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd, uint_fast8_t* __restrict__ ILtype,
	const uint_fast32_t IN, const float m_Gcritn, const float m_Gcritt, const float b_r)
{	
	uint_fast32_t kdx = blockIdx.x * blockDim.x + threadIdx.x;
	while (kdx < IN)
	{
		if (ILtype[kdx] == 1)
		{
			//float rijm = norm3df(Rij[kdx].x, Rij[kdx].y, Rij[kdx].z);
			float _1d_areaij = __frcp_rn(b_r * b_r * MCf_pi);
			//float oijnm = (rijm - __frcp_rn(_1d_iL[kdx]));
			float fn = Fijn[kdx].x * Rij[kdx].x + Fijn[kdx].y * Rij[kdx].y + Fijn[kdx].z * Rij[kdx].z;
			float Sn = norm3df(Fijn[kdx].x, Fijn[kdx].y, Fijn[kdx].z) * _1d_areaij;

			if (fn < 0)
				Sn *= -1.0f;

			float _1d_Iij = __frcp_rn(MCf_pi * b_r * b_r * b_r * b_r * 0.25f);
			float St = norm3df(Mijt[kdx].x, Mijt[kdx].y, Mijt[kdx].z) * b_r * _1d_Iij;
			float MSn = norm3df(Mijn[kdx].x, Mijn[kdx].y, Mijn[kdx].z) * 0.5f * b_r * _1d_Iij;
			float MSt = norm3df(Fijt[kdx].x, Fijt[kdx].y, Fijt[kdx].z) * _1d_areaij;
			if ((Sn + St > m_Gcritn) || (MSn + MSt > m_Gcritt))
			{
				ILtype[kdx] = 0;
				//Fijn[kdx].x = 0; Fijn[kdx].y = 0; Fijn[kdx].z = 0;
				//Fijt[kdx].x = 0; Fijt[kdx].y = 0; Fijt[kdx].z = 0;
				Oijt[kdx].x = 0; Oijt[kdx].y = 0; Oijt[kdx].z = 0;
				//Mijn[kdx].x = 0; Mijn[kdx].y = 0; Mijn[kdx].z = 0;
				//Mijt[kdx].x = 0; Mijt[kdx].y = 0; Mijt[kdx].z = 0;
				//Mijadd[kdx].x = 0; Mijadd[kdx].y = 0; Mijadd[kdx].z = 0;
			}
		}			
		kdx += blockDim.x * gridDim.x;
	}	
}
