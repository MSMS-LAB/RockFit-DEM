#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "md_data_types.h"
void DetermineConjGradient();
void h_CreateCylinderSample(mp_mdparameters_data& mpMDP, md_task_data& mdTD, uint_fast32_t isample, char namepart[] = "");
void h_LoadCylinderSample(mp_mdparameters_data& mpMDP, uint_fast32_t isample);
void h_LoadCylinderSample(mp_mdparameters_data& mpMDP, uint_fast32_t isample, char* filename);

void h_CalculationUniaxialCompression(md_task_data& mdTD, float* h_R);
void h_CalculationBrazilTest(md_task_data& mdTD, float* h_R);
