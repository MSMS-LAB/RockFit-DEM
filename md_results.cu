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
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <math.h>

void ResultsUInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, result_data &R, compression_data& Compress)
{	
	R.N = 2 * A.bloks;
	R.bloks = A.bloks;	
	R.stepsave = 0;
	R.sz0 = Compress.size.z;
	HANDLE_ERROR(cudaMalloc((void**)&R.d_FL, R.N * sizeof(float)));
	R.h_FL = (float*)malloc(R.N * sizeof(float));
	R.h_sFL = (double*)malloc(3 * R.Nsave * sizeof(double));
}

void ResultsUDelete(result_data& R)
{
	if (R.d_FL != nullptr) { cudaFree(R.d_FL); R.d_FL = nullptr; }
	if (R.h_FL != nullptr) { free(R.h_FL); R.h_FL = nullptr; }
	if (R.h_sFL != nullptr) { free(R.h_sFL); R.h_sFL = nullptr; }
}

void SumForcesULoading(result_data& R, compression_data & Compress, uint_fast32_t n)
{
	HANDLE_ERROR(cudaMemcpy(R.h_FL, R.d_FL, R.N * sizeof(float), cudaMemcpyDeviceToHost));
	uint_fast32_t ir;
	R.sFL[0] = 0;	
	for (ir = 0; ir < R.bloks; ++ir)
	{
		R.sFL[0] += R.h_FL[ir];
	}
	R.sFL[1] = 0;
	for (ir = R.bloks; ir < 2*R.bloks; ++ir)
	{
		R.sFL[1] += R.h_FL[ir];
	}
	if (n % R.dNsave == 0)
	{
		R.h_sFL[3 * R.stepsave] = (Compress.size_d.z - R.sz0) / R.sz0;
		R.h_sFL[3 * R.stepsave + 1] = R.sFL[0] * Compress._1d_Area;
		R.h_sFL[3 * R.stepsave + 2] = R.sFL[1] * Compress._1d_Area;
		++R.stepsave;
	}
	Compress.Stress = 0.5 * (fabs(R.sFL[0]) + fabs(R.sFL[1])) * Compress._1d_Area;
	if (!Compress.Loading && Compress.Stress > Compress.MinStress)
	{
		Compress.Loading = true;
	}
	if (Compress.FractureStress < Compress.Stress)
	{
		Compress.FractureStress = Compress.Stress;
		Compress.FractureStrain = (R.sz0 - Compress.size_d.z) / R.sz0;
	}
	if (Compress.Loading)
	{
		//std::cerr << "F " << Compress.FractureStress << " " << Compress.Stress << "\n";		
		if (Compress.Stress < Compress.ZeroStress)
		{
			Compress.Fracture = true;
		}
	}
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void SumForcesBLoading(result_data& R, compression_data& Compress, uint_fast32_t n)
{
	HANDLE_ERROR(cudaMemcpy(R.h_FL, R.d_FL, R.N * sizeof(float), cudaMemcpyDeviceToHost));
	uint_fast32_t ir;
	R.sFL[0] = 0;
	for (ir = 0; ir < R.bloks; ++ir)
	{
		R.sFL[0] += R.h_FL[ir];
	}
	R.sFL[1] = 0;
	for (ir = R.bloks; ir < 2 * R.bloks; ++ir)
	{
		R.sFL[1] += R.h_FL[ir];
	}
	if (n % R.dNsave == 0)
	{
		R.h_sFL[3 * R.stepsave] = (Compress.size_d.z - R.sz0) / R.sz0;
		R.h_sFL[3 * R.stepsave + 1] = R.sFL[0] * Compress._1d_Area;
		R.h_sFL[3 * R.stepsave + 2] = R.sFL[1] * Compress._1d_Area;
		++R.stepsave;
	}
	Compress.Stress = 0.5 * (fabs(R.sFL[0]) + fabs(R.sFL[1])) * Compress._1d_Area;
	if (!Compress.Loading && Compress.Stress > Compress.MinStress)
	{
		Compress.Loading = true;
	}
	if (Compress.FractureStress < Compress.Stress)
	{
		Compress.FractureStress = Compress.Stress;
		Compress.FractureStrain = (R.sz0 - Compress.size_d.z) / R.sz0;
	}
	if (Compress.Loading)
	{
		//std::cerr << "F " << Compress.FractureStress << " " << Compress.Stress << "\n";
		
		if (Compress.Stress < Compress.ZeroStress)
		{
			Compress.Fracture = true;
		}
	}
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

void SaveSumForcesLoading(result_data& R, char* Name)
{
	std::ofstream file;
	file.open(Name, std::ios::out);
	file << "EpsZ StressZ_bottom StressZ_top\n";
	uint_fast32_t i;
	for (i = 0; i < R.stepsave; ++i)
	{
		file << R.h_sFL[3 * i] << " " << R.h_sFL[3 * i + 1] * stress_const << " " << R.h_sFL[3 * i + 2] * stress_const << "\n";
	}	
	file.close();
	//std::cerr << "R " << R.bloks << " " << R.sFL[0] << " " << R.sFL[1] << "\n";
}

