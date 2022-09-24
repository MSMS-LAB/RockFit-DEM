#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "md_data_types.h"
#define pre_debugtest


void initArrays(additional_data &A, particle_data& P, cell_data& C, interaction_list_data& IL, result_data& R);
void deleteArrays(additional_data &A, particle_data& P, cell_data& C, interaction_list_data& IL, result_data& R);
void free_particle_data(particle_data& P);
void free_interaction_list_data(interaction_list_data& IL);

void generateParticles(particle_data& P, additional_data& A, sample_data& S);
__global__ void d_SetParticlesInParallelepiped(float* __restrict__ R, float* __restrict__ V, uint_fast32_t N, const float3 c, const float3 L, const float vm);
void generateVelocities(particle_data& P, additional_data& A, sample_data& S);


void CellDistributionInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C);
void CellDistribution(particle_data& P, additional_data& A, sample_data& S, cell_data& C);
__global__ void d_FillIndex(uint_fast32_t* __restrict__ CI, unsigned  int N);
__global__ void d_CalculateCellIndex(const float* __restrict__ R, uint_fast32_t N, uint_fast32_t* __restrict__ CI, float _1d_a, uint3 cN);
__global__ void d_DetermineCellPointer(const uint_fast32_t* __restrict__ CIs, uint_fast32_t* __restrict__ pnC, uint_fast32_t N, uint_fast32_t CN);

void InteractionListInit(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL);
void InteractionListConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);

__global__ void d_ConstructInteractionList(const float* __restrict__ R, uint_fast32_t N, const uint_fast32_t* __restrict__ CI, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t* __restrict__ pnC, uint_fast32_t* __restrict__ IL, uint_fast32_t Iin, float a, float _1d_a, float aacut, uint3 cN, uint_fast32_t CN);
__device__ void addlink(const float* __restrict__ R, const uint_fast32_t N, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t idx, const uint_fast32_t nindx, const uint_fast32_t jindx, uint_fast32_t* __restrict__ IL, uint_fast32_t Iin, float aacut, uint_fast32_t& nkndx);

void RenewInteractionList_full(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);

__global__ void d_CalculateForces(const float* __restrict__ R, float* __restrict__ F, uint_fast32_t N, const uint_fast32_t* __restrict__ IL, float* __restrict__ Rijm, const uint_fast32_t IonP, const float D, const float aa);
__global__ void d_CalculateForcesLJ(const float* __restrict__ R, float* __restrict__ F, uint_fast32_t N, const uint_fast32_t* __restrict__ IL, const uint_fast32_t IonP, const float D, const float a2, const float _1d_a2);
__global__ void d_CalculateIncrements(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float _1d_Mass_m_dt, const float dt);
__global__ void d_CylinderRestriction(float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float3 axis, const float3 center, const float R0, const float H0);
__global__ void d_ParallelepipedRestriction(float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float3 center, const float3 sized2);
__global__ void d_CylinderRestrictionZ(float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float3 center, const float R0, const float H0);
__global__ void d_CylinderBorderPush(float* __restrict__ Fcoeff, const float Fcoeffborder, float* __restrict__ R, const uint_fast32_t N, const float3 center, const float R0, const float H0);
__global__ void d_CalculateIncrementsViscos(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const uint_fast32_t N, const float _1d_Mass_m_dt, const float dt, const float vis);

void CalculateGPUStepsContractRelaxFIRE(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po, firerelax_data& Fire);
void SetFIREData(particle_data& P, potential_data& Po, firerelax_data& Fire);
__global__ void d_CalculateIncrementsFIRE(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float _1d_Mass_m_dt, const float dt, const float F_alpha);
__global__ void d_CalculateDecrementsHalfStepFIRE(const float* __restrict__ V, float* __restrict__ R, const uint_fast32_t N, const float dt_d2);
__global__ void d_FdotVEntire(const float* __restrict__ V, const float* __restrict__ F, float* FdotV, const uint_fast32_t N);

void RenewInteractionList_New(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);
void InteractionListReConstruct(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);
__global__ void d_ReConstructInteractionList(const float* __restrict__ R, uint_fast32_t N, const uint_fast32_t* __restrict__ CI, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t* __restrict__ pnC, uint_fast32_t* __restrict__ IL, uint_fast32_t IonP, float a, float _1d_a, float aacut, uint3 cN, uint_fast32_t CN);
__device__ void addnewlink(const float* __restrict__ R, const uint_fast32_t& N, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t& idx, const uint_fast32_t& nindx, const uint_fast32_t& jindx, uint_fast32_t* __restrict__ IL, uint_fast32_t& IonP, const float& aacut, uint_fast32_t& nkndx);
__device__ uint_fast32_t deletelinks(const float* __restrict__ R, const uint_fast32_t& N, const uint_fast32_t& idx, uint_fast32_t* __restrict__ IL, uint_fast32_t& IonP, float& aacut, uint_fast32_t& nkndx);

__global__ void d_braziltest_simple(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint_fast32_t N, const float3 c, const float RR, const float Yt, const float Yb, const float Ytr, const float Ybr, const float vt, const float Zcut);
__global__ void d_braziltest_simple2(float* __restrict__ R, float* V, float* F, float* __restrict__ FL, const uint_fast32_t N, const float3 c, const float RR, const float Yt, const float Yb, const float Ytr, const float Ybr, const float vt, const float Zcut);

__global__ void d_UniaxialCompression_simple(const float* __restrict__ R, const float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint_fast32_t N, const float3 c, const float RR, const float Hd2, const float C, const float mu, const float Zcut);
__global__ void d_UniaxialCompression2_simple(float* __restrict__ R, const float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint_fast32_t N, const float3 c, const float RR, const float Hd2, const float C, const float mu, const float Zcut);
__global__ void d_UniaxialCompression3_simple(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, float* __restrict__ FL, const uint_fast32_t N, const float3 c, const float RR, const float Zt, const float Zb, const float Ztr, const float Zbr, const float vt, const float Zcut);
__global__ void d_ParallelepipedCutRestriction(float* __restrict__ R, float* __restrict__ V, float* __restrict__ F, const uint_fast32_t N, const float3 center, const float3 sized2, const float3 hp);

__global__ void d_ConstructBoundInteractions(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float aa_create, const float b_r);
__global__ void d_CalculateForcesDEM(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt,
	const float m_E, const float m_G, const float b_r,
	const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
//__device__ void dd_CalculateForceElastic(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, float* __restrict__ F, float* __restrict__ M,
//	const float* __restrict__ b_r, const float* __restrict__ InitialLength, const float* __restrict__ AxialMoment,
//	float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt,
//	float3* __restrict__ Fij, const uint_fast32_t& N, const uint_fast32_t& idx, const uint_fast32_t& jdx, const uint_fast32_t& kdx,
//	const float& dt, const float& m_E, const float& m_G, const float& m_Gcritn, const float& m_Gcritt, uint_fast8_t& type);
__device__ void dd_CalculateForceElastic(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W, float* __restrict__ F, float* __restrict__ M,
	const float* __restrict__ b_r, const float* __restrict__ _1d_iL, const float* __restrict__ AxialMoment,
	float3& rij_p, float3& oijt, float3& mijn, float3& mijt, float3& fijn, float3& fijt, float3& Munsym, 
	const uint_fast32_t& N, const uint_fast32_t& idx, const uint_fast32_t& jdx, const uint_fast32_t& kdx, const float& dt, const float& m_E, const float& m_G);

__device__ void dd_CalculateForceParticleParticle(const float* __restrict__ R, const float* __restrict__ V, const float* __restrict__ W,
	float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt,
	float3* __restrict__ Fij, const uint_fast32_t& N, const uint_fast32_t& idx, const uint_fast32_t& jdx, const uint_fast32_t& kdx,
	const float& dt, const float& m_E, const float& m_G, const float& p_A, const float& m_mu, const float& m_muroll, const float& Rp, const float& Mp);
__global__ void d_CalculateIncrementsDEM(const float* __restrict__ F, float* __restrict__ V, float* __restrict__ R,
	const float* __restrict__ M, float* __restrict__ W,
	const uint_fast32_t N, const float _1d_Mass_m_dt, const float dt, const float _1d_I_m_dt);
void SaveAllData(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL, potential_data &Po);
void LoadAllData(particle_data &P, cell_data &C, sample_data &S, additional_data &A, interaction_list_data &IL, potential_data &Po);

void write_sample_data(std::ofstream& file, sample_data& S);
void read_sample_data(std::ifstream& file, sample_data& S);
void write_particle_data(std::ofstream& file, particle_data& P);
void read_particle_data(std::ifstream& file, particle_data& P);
void write_interaction_list_data(std::ofstream& file, interaction_list_data& IL);
void read_interaction_list_data(std::ifstream& file, interaction_list_data& P);

void SaveLammpsDATASimple(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po,
	char* Name, bool flagv = true);


void h_CalculateForceElastic(const float* R, const float* V, const float* W, float* F, float* M,
	const float* b_r, const float* _1d_iL, const float* AxialMoment, float3* Rij, float3* Oijt, float3* Mijn, float3* Mijt, float3* Fij,
	const uint_fast32_t N, const interaction_list_data& IL, const particle_data& P, const float dt, const float m_E, const float m_G, const float m_Gcritn, const float m_Gcritt);

__device__ void dd_Calculate_rijm_nij(const float* __restrict__ R, const uint_fast32_t& idx, const uint_fast32_t& jdx, const uint_fast32_t& N, float3& nij, float& rijm);

__global__ void d_SumUpForcesDEM(const uint_fast32_t* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const	uint_fast32_t N, const uint_fast32_t IonP);
__global__ void d_SumUpForcesDEMViscos(const uint_fast32_t* __restrict__ IL, const float3* __restrict__ Fijn, const float3* __restrict__ Fijt, const float3* __restrict__ Mijn, const float3* __restrict__ Mijt, const float3* __restrict__ Mijadd,
	float* __restrict__ F, float* __restrict__ M, const float* __restrict__ V, const float* __restrict__ W, const	uint_fast32_t N, const uint_fast32_t IonP, const float nuV, const float nuW);


__global__ void d_CheckBreakConditionDEM(const uint_fast32_t* __restrict__ IL, const float* __restrict__ _1d_iL,
	const float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	uint_fast8_t* __restrict__ ILtype, const uint_fast32_t IN, const float m_Gcritn, const float m_Gcritt, const float b_r);

__global__ void d_CutCylinderSpecimen_simple(float* __restrict__ R, float* __restrict__ V, const uint_fast32_t N, const float3 center, const float RR0, const float H0, const float3 hidenpoint);
__global__ void d_DeleteFarLinks(const float* __restrict__ R, uint_fast32_t N, uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype, float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd, uint_fast32_t IonP, const float aacut);


__global__ void d_CalculateForcesDEM_1(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
__global__ void d_CalculateForcesDEM_2(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt,
	float3* __restrict__ Fijn, float3* __restrict__ Fijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float3* __restrict__ Mijadd,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt, const float m_E, const float m_G, const float b_r);

void RenewInteractionList_BPM(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL);
void InteractionListReConstructBPM(particle_data& P, additional_data& A, sample_data& S, interaction_list_data& IL, cell_data& C);
__global__ void d_ReConstructInteractionListbpm(const float* __restrict__ R, uint_fast32_t N, const uint_fast32_t* __restrict__ CI, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t* __restrict__ pnC, uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype, uint_fast32_t IonP, float a, float _1d_a, float aacut, uint3 cN, uint_fast32_t CN);
__device__ void addnewlinkbpm(const float* __restrict__ R, const uint_fast32_t& N, const uint_fast32_t* __restrict__ CIs, const uint_fast32_t& idx, const uint_fast32_t& nindx, const uint_fast32_t& jindx, uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype, uint_fast32_t& IonP, float& aacut, uint_fast32_t& nkndx);

void ResultsUInit(particle_data& P, additional_data& A, sample_data& S, cell_data& C, result_data& R, compression_data& Compress);
void ResultsUDelete(result_data& R);
void SumForcesULoading(result_data& R, compression_data& Compress, uint_fast32_t n);
void SaveSumForcesLoading(result_data& R, char* Name);

void h_CheckInteractions(particle_data& P, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL);
void CheckDATA(particle_data& P, potential_data& Po, cell_data& C, additional_data& A, sample_data& S, interaction_list_data& IL);

void ReadParticlesCSV(particle_data& P, cell_data& C, sample_data& S, additional_data& A, interaction_list_data& IL, potential_data& Po, char* Name, uint_fast32_t N);

__global__ void d_CalculateForcesDEM_11(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float* __restrict__ F, float* __restrict__ M,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);
__global__ void d_CalculateForcesDEM_12(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt, const float m_E, const float m_G, const float b_r);

__global__ void d_CalculateForcesDEM_21(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float* __restrict__ F, float* __restrict__ M,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt,
	const float m_E, const float m_G, const float p_A,
	const float m_mu, const float m_muroll, const float mP, const float rP);

__global__ void d_CalculateForcesDEM_22(const float* __restrict__ R, const float* __restrict__ V,
	const float* __restrict__ W, const	uint_fast32_t* __restrict__ IL, uint_fast8_t* __restrict__ ILtype,
	const float* __restrict__ _1d_iL, float3* __restrict__ Rij, float3* __restrict__ Oijt, float3* __restrict__ Mijn, float3* __restrict__ Mijt, float* __restrict__ F, float* __restrict__ M,
	const	uint_fast32_t N, const uint_fast32_t IonP, const float dt, const float m_E, const float m_G, const float b_r, const float m_Gcritn, const float m_Gcritt);


