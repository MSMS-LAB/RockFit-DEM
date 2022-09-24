#include "md_phys_constants.h"
#include "md_data_types.h"
#include "md.h"
#include <stdint.h>
#include <iostream>
#include <fstream>
#include "pcuda_helper.h"

void ReadParticlesCSV(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po, char* Name, uint_fast32_t N)
{
	std::ifstream file;
	file.open(Name, std::ios::in);
	P.N = N;
	A.bloks = ceil(P.N / (SMEMDIM)) + 1;
	HANDLE_ERROR(cudaMalloc((void**)&P.d_R, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_V, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_F, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_M, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_W, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&P.d_Q, P.N * sizeof(float4)));

	HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_F, 0, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_W, 0, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_M, 0, 3 * P.N * sizeof(float)));
	HANDLE_ERROR(cudaMemset((void*)P.d_Q, 0, P.N * sizeof(float4)));
	//HANDLE_ERROR(cudaMalloc((void**)&P.d_U, 3 * P.N * sizeof(float)));
	P.h_F = (float*)malloc(3 * P.N * sizeof(float));
	P.h_V = (float*)malloc(3 * P.N * sizeof(float));
	P.h_R = (float*)malloc(3 * P.N * sizeof(float));
	P.h_M = (float*)malloc(3 * P.N * sizeof(float));
	P.h_W = (float*)malloc(3 * P.N * sizeof(float));
	P.h_Q = (float4*)malloc(P.N * sizeof(float4));
	uint_fast32_t i;
	double3 r;
	for (i = 0; i < P.N; ++i)
	{
		file >> r.x >> r.y >> r.z;
		P.h_R[i] = r.x / length_const + S.center.x;
		P.h_R[i + P.N] = r.y / length_const + S.center.y;
		P.h_R[i + 2 * P.N] = r.z / length_const + S.center.z;
		//std::cerr << "P " << i << " " << r.x << " " << r.y << " " << r.z << "\n";
		//<< " " << 0.5 * Po.m * P.h_V[i] * P.h_V[i] << " " << 0.5 * Po.m * P.h_V[i + P.N] * P.h_V[i + P.N]//<<"\n";
		//<< " " << P.h_V[i] << " " << P.h_V[i + P.N]
		//<< " " << P.h_F[i] << " " << P.h_F[i + P.N] << "\n";
	}
	file.close();

	HANDLE_ERROR(cudaMemcpy(P.d_R, P.h_R, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
	S.A_real.x = 1e20; S.A_real.y = 1e20; S.A_real.z = 1e20;
	S.B_real.x = -1e20; S.B_real.y = -1e20; S.B_real.z = -1e20;
	for (int i = 0; i < P.N; ++i)
	{
		if (P.h_R[i] < S.A_real.x)
			S.A_real.x = P.h_R[i];
		if (P.h_R[i] > S.B_real.x)
			S.B_real.x = P.h_R[i];

		if (P.h_R[i + P.N] < S.A_real.y)
			S.A_real.y = P.h_R[i + P.N];
		if (P.h_R[i + P.N] > S.B_real.y)
			S.B_real.y = P.h_R[i + P.N];

		if (P.h_R[i + 2 * P.N] < S.A_real.z)
			S.A_real.z = P.h_R[i + 2 * P.N];
		if (P.h_R[i + 2 * P.N] > S.B_real.z)
			S.B_real.z = P.h_R[i + 2 * P.N];
	}
	S.size_real.x = S.B_real.x - S.A_real.x; S.size_real.y = S.B_real.y - S.A_real.y; S.size_real.z = S.B_real.z - S.A_real.z;
	S.center_real.x = 0.5 * (S.B_real.x + S.A_real.x); S.center_real.y = 0.5 * (S.B_real.y + S.A_real.y); S.center_real.z = 0.5 * (S.B_real.z + S.A_real.z);
	std::cerr << "AB real " << S.A_real.x << " " << S.A_real.y << " " << S.A_real.z << " " << S.B_real.x << " " << S.B_real.y << " " << S.B_real.z << "\n";
	//std::cerr << "Save PBM "<<"\n";
	//std::cerr << "LL " << P.N << " " << pnid2 << " | " << P.N + pnid2 << " " << P.NI << " " << j << "\n";
}
