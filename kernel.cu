#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include "pcuda_helper.h"
#include "md_data_types.h"
#include "md_math_constants.h"
#include "md_phys_constants.h"
#include "md.h"
#include "mp.h"
#include <chrono>
float normm(float3 r)
{
    return sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
}

float normm(float* r, uint_fast32_t i, uint_fast32_t N)
{
    return sqrt(r[i] * r[i] + r[i + N] * r[i + N] + r[i + 2 * N] * r[i + 2 * N]);
}
int main()
{

    DetermineConjGradient();
    std::cin.get();
    particle_data P;
    cell_data C;
    sample_data S;
    additional_data A;
    interaction_list_data IL;
    potential_data Po;
    result_data R;
    IL.IonP = 30;   
    double CritCorr = 1.6, ECorr = 1.3834;//1.6857
    Po.m_nu = 0.247;
    Po.m_E = ECorr * 1.95255e+11 / stress_const;
    Po.m_G = 0.5 * Po.m_E / (1.0 + Po.m_nu);
    Po.m_Ec = CritCorr * 5.1447e+08 / stress_const;
    Po.m_Gc = CritCorr * 3.7131e+08 / stress_const;
    Po.m_ro = 2610 / density_const;
    Po.b_r = 0.5 * 0.000330015 / length_const;
    Po.b_S = MC_pi * Po.b_r * Po.b_r;
    Po.p_r = 0.5 * 0.001 / length_const;
    Po.p_V = 4.0 * MC_1d3 * MC_pi * Po.p_r * Po.p_r * Po.p_r;
    Po.p_m = Po.m_ro * Po.p_V;
    Po.b_d = 2.0 * Po.p_r + 5.93169e-05 / length_const;
    Po.b_dd = Po.b_d * Po.b_d;
    
    //std::cerr << "Pm " << Po.m << "\n"; std::cin.get();
    Po.p_1d_m = 1.0 / Po.p_m;
    Po.p_I = 0.4 * Po.p_m * Po.p_r * Po.p_r;
    Po.p_e = 0.6;
    Po.p_mu = 0.45;
    Po.p_mur = 0.05;
    //Po.m_E = Po.m_E;
    //Po.m_G = Po.m_G;
    Po.hm_E = 0.5 * Po.m_E / (1.0 - Po.m_nu * Po.m_nu);
    Po.hm_G = 0.5 * Po.m_G / (2.0 - Po.m_nu);
    //Po.particlematerial_Poisson = Po.boundmaterial_Poisson;
    Po.p_A = log(Po.p_e) / sqrt(MC_pi * MC_pi + log(Po.p_e) * log(Po.p_e));
    std::cerr << "Material " << "E=" << Po.m_E << " G=" << Po.m_G << " Ecrit=" << Po.m_Ec << " Gcrit=" << Po.m_Gc << " density=" << Po.m_ro << "\n"
        << "Particle" << " Radius=" << Po.p_r << " " << " Volume=" << Po.p_V << " Mass=" << Po.p_m << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A
        << " HM_E=" << Po.hm_E << " HM_G=" << Po.hm_G << " Friction=" << Po.p_mu << " RollFriction=" << Po.p_mur << "\n"
        << "Bound" << " Radius=" << Po.b_r << " " << " Area=" << Po.b_S << " Distance=" << Po.b_d << " Inertia=" << Po.p_I << " e=" << Po.p_e << " Alpha=" << Po.p_A << "\n";

    std::cerr << "Alpha " << Po.p_A << " " << Po.p_m << " " << Po.p_r << " " << -1.8257 * Po.p_A * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) <<" "<<-2.0*sqrt(5.0/6.0)* Po.p_e * sqrt(2.0f * Po.hm_E * 0.5f * Po.p_m) << "\n";
    
    std::cerr << "Test " << (Po.p_r - 2.0f * Po.p_r) * 2.0f * Po.hm_E * sqrt(Po.p_r * fabsf(Po.p_r - 0.5f * Po.p_r)) * MCf_2d3 << " " << Po.b_r * Po.b_r * MCf_pi * Po.m_E * (Po.p_r - 2.0 * Po.p_r) / (2.0 * Po.p_r) << "\n";
    std::cerr << "Test1 " << Po.m_Ec / MC_pi * Po.b_r * Po.b_r * Po.m_E << "\n";
    //std::cerr << "Phm " << Po.m_E << " " << Po.m_G << " " << Po.hm_E << " " << Po.hm_G << "\n";
    //std::cin.get();
    
    Po.nuV = 1e-5;
    Po.nuW = 1e-3;
    Po.a_farcut = 2.9 * Po.p_r;
    Po.aa_farcut = Po.a_farcut * Po.a_farcut;
    C.a  = 2.0 * Po.a_farcut;
    IL.acut = Po.a_farcut;

    S.spacesize.x = 0.04 / length_const;
    S.spacesize.y = 0.04 / length_const;
    S.spacesize.z = 0.06 / length_const;
    S.spacesized2.x = 0.5 * S.spacesize.x; S.spacesized2.y = 0.5 * S.spacesize.y; S.spacesized2.z = 0.5 * S.spacesize.z;
    S.L.x = 0.025 / length_const - 2.0 * Po.p_r;
    S.L.y = 0.025 / length_const - 2.0 * Po.p_r;
    S.L.z = 0.05 / length_const - 2.0 * Po.p_r;
   
    S.A.x = 1.01 * C.a; S.A.y = 1.01 * C.a; S.A.z = 1.01 * C.a;
    S.B.x = S.A.x + S.spacesize.x; S.B.y = S.A.y + S.spacesize.y; S.B.z = S.A.z + S.spacesize.z;
    S.axis.x = 0; S.axis.y = 0; S.axis.z = 1.0f;
    S.center.x = 0.5 * (S.A.x + S.B.x); S.center.y = 0.5 * (S.A.y + S.B.y); S.center.z = 0.5 * (S.A.z + S.B.z);
    S.size = S.L;
    S.sized2.x = 0.5 * S.L.x; S.sized2.y = 0.5 * S.L.y; S.sized2.z = 0.5 * S.L.z;
    S.R0 = 0.5 * S.L.x; S.H0 = 0.5 * S.L.z;
    std::cerr << "Sample " << S.L.x << " " << S.L.y << " " << S.L.z << " | " << S.B.x << " " << S.B.y << " " << S.B.z << "\n";
    Po.Trayleigh = MC_pi * Po.p_r * sqrt(2.0 * Po.m_ro * (1.0 + Po.m_nu)) / (sqrt(Po.m_E) * (0.163 * Po.m_nu + 0.8766));
    double tmp[2] = { Po.m_E * Po.b_S / (2.0 * Po.p_r), Po.m_G * Po.b_S / (2.0 * Po.p_r) };
    if (tmp[0] > tmp[1])
        Po.Tsim = 2.0 * sqrt(Po.p_m / tmp[0]);
    else 
        Po.Tsim = 2.0 * sqrt(Po.p_m / tmp[1]);
    
    if (Po.Trayleigh < Po.Tsim)
        Po.dt = 0.1 * Po.Trayleigh;
    else
        Po.dt = 0.1 * Po.Tsim;
    Po.dt *= 0.1*10;
    //Po.dt = 2e-8;
    Po.D = 1e-15*Po.m_E * Po.b_S / (2.0 * Po.p_r);
    std::cerr << "Po " << Po.D << "\n";
    Po.a = (2.0 + 0*0.1e-5) * Po.p_r;
    Po._1d_a = 1.0 / Po.a;
    Po.aa = Po.a * Po.a;
    
    Po.dt_d_m = Po.p_1d_m * Po.dt;
    Po.dt_d_I = Po.dt / Po.p_I;
    std::cerr << "Param " << Po.dt << " " << Po.Trayleigh << " " << Po.Tsim << " | " << Po.p_m << " " << Po.dt_d_m << " " << Po.dt_d_I << "\n";
    //std::cin.get();

    P.N = 3*228;
    //S.Vext = (S.L.x + 2.0 * Po.p_r) * (S.L.y + 2.0 * Po.p_r) * (S.L.z + 2.0 * Po.p_r);
    S.Vext = MC_pi * (0.5 * S.L.x + Po.p_r) * (0.5 * S.L.x + Po.p_r) * (S.L.z + 2.0 * Po.p_r);
    P.N = (1.0 - 0.39) * S.Vext / Po.p_V;
    //P.N = 6;
    //P.N = 0.8 * MC_pi * S.R0 * S.R0 * 2.0 * S.H0 / Po.pV;
    std::cerr << "ParamV " << P.N << " " << Po.p_V * P.N * volume_const << " " << S.Vext * volume_const << " " << Po.p_V * P.N / S.Vext << " " << "\n";
    //std::cin.get();
    
    char filename[256] = "";
    initArrays(A, P, C, IL, R);
    
    S.Vgenmax = 1e-3 * Po.p_r / Po.dt;
    Po.vis = 1.0e-5;
    Po.vism = Po.vis * Po.p_m;
    
    /*generateParticles(P, A, S);
    InteractionListInit(P, A, S, IL);
    CellDistributionInit(P, A, S, C);   
    firerelax_data Fire;
    SetFIREData(P, Po, Fire);
    Fire.MaxStepsRelaxation = 1000000;
    CalculateGPUStepsContractRelaxFIRE(P, C, S, A, IL, Po, Fire);
    //std::cerr << "Interaction listA " << IL.N << "\n";
    //cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < IL.N; ++i)
    //{
    //    if (i % IL.IonP == 0)std::cerr << "\n" << i / IL.IonP;
	//			if (IL.h_IL[i] != UINT_FAST32_MAX)
	//				std::cerr << " " << IL.h_IL[i] << "&" << int(IL.h_ILtype[i]);									
	//			else if (IL.h_IL[i] < UINT_FAST32_MAX && IL.h_IL[i] >= P.N)std::cerr << "!!!!";
	//			else if (IL.h_IL[i] == UINT_FAST32_MAX)
	//			{
	//				std::cerr << " " << "M";
	//				continue;
	//			}	
    //}
    SaveAllData(P, C, S, A, IL, Po);
    std::cin.get();/**/
    //deleteArrays(A, P, C, IL);/**/
    //LoadAllData(P, C, S, A, IL, Po);
    //sprintf(filename, "./result/ExprotedData.csv", 0);
    //ReadParticlesCSV(P, C, S, A, IL, Po, filename, 3457);
    sprintf(filename, "./result/ExprotedData_1.csv", 0);
    ReadParticlesCSV(P, C, S, A, IL, Po, filename, 28593);
    InteractionListInit(P, A, S, IL);
    CellDistributionInit(P, A, S, C);

    //HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    //sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", 0);
    //SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
    //std::cin.get();
    //Po.dt *= 0.25;
    

    compression_data Compress;
    Compress.size_d.x = 1.0 * S.size_real.x; Compress.size_d.y = 1.0 * S.size_real.y; Compress.size_d.z = 1.0 * S.size_real.z;
    Compress.size.x = Compress.size_d.x; Compress.size.y = Compress.size_d.y; Compress.size.z = Compress.size_d.z;
    Compress.sized2.x = 0.5 * Compress.size_d.x; Compress.sized2.y = 0.5 * Compress.size_d.y; Compress.sized2.z = 0.5 * Compress.size_d.z;
    Compress.center_d.x = S.center_real.x; Compress.center_d.y = S.center_real.y; Compress.center_d.z = S.center_real.z;
    Compress.Bottom = Compress.center_d.z - 0.5 * Compress.size_d.z;
    Compress.BottomR = Compress.Bottom + Po.p_r;
    Compress.Top = Compress.center_d.z + 0.5 * Compress.size_d.z;
    Compress.TopR = Compress.Top - Po.p_r;
    Compress.center.x = Compress.center_d.x; Compress.center.y = Compress.center_d.y; Compress.center.z = Compress.center_d.z;
    Compress.Hd2 = Compress.sized2.z;
    Compress.R = 1.5*Compress.sized2.x;
    Compress.RR = Compress.R * Compress.R;
    Compress.Area = MC_pi * (Compress.sized2.x + Po.p_r) * (Compress.sized2.x + Po.p_r);
    Compress._1d_Area = 1.0 / Compress.Area;
    Compress.C = Po.m_E * Po.b_S / (4.0 * Po.p_r);
    Compress.mu = Po.p_mu;
    Compress.V = -0*0.02 / velocity_const;

    R.Nsave = 1000000;
    R.dNsave = 1;
    ResultsUInit(P, A, S, C, R, Compress);

    HANDLE_ERROR(cudaMemset((void*)P.d_V, 0, 3 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)P.d_W, 0, 3 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)P.d_M, 0, 3 * P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)P.d_Q, 0, P.N * sizeof(float4)));

    HANDLE_ERROR(cudaMemset((void*)IL.d_1d_iL, 0, IL.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_rij, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Oijt, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Fijn, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Fijt, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Mijn, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Mijt, 0, IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)IL.d_Mijadd, 0, IL.N * sizeof(float3)));
    std::cerr << "T " << P.N << " " << IL.N << " " << IL.IonP << " " << A.bloks << " " << SMEMDIM << "\n";
    
    std::cerr << "Cut " << Compress.Top << " " << Compress.Bottom << " | " << S.B_real.z << " " << S.A_real.z << " " << Po.p_r << " | " <<Compress.Area*length_const*length_const << "\n";
    //d_CutCylinderSpecimen_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.N, Compress.center, Compress.sized2.x* Compress.sized2.x, Compress.Hd2, S.hidenpoint);
    //std::cin.get();
    //CellDistributionInit(P, A, S, C);
    std::cerr << "Bound " << Po.b_d << " " << Po.b_dd << " " << sqrt(Po.b_dd) << "\n";
    RenewInteractionList_full(P, C, S, A, IL);
    d_ConstructBoundInteractions << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_W,
        IL.d_IL, IL.d_ILtype, IL.d_1d_iL, IL.d_rij, IL.d_Oijt, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, P.N, IL.IonP, Po.b_dd, Po.b_r);
    
    //std::cin.get();
    //std::cerr << "AAAA\n";
    HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
    uint_fast32_t i, j, ILn = 0, ILn0 = 0, Pn = 0;
    for (i = 0; i < IL.N; ++i)
    {
        j = i / IL.IonP;
        if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 1)
            ++ILn;
        if (IL.h_IL[i] < P.N && IL.h_ILtype[i] == 0)
            ++ILn0;
    }
    for (i = 0; i < P.N; ++i)    
        if (P.h_R[i + 2 * P.N] < S.center.z + S.spacesized2.z)
            ++Pn;
    
    std::cerr << "Interactions Number " << ILn << " " << ILn0 << " " << Pn << "\n";
    //std::cin.get();
    double Vm = 0.05 * 0.0125 * 0.0125 * MC_pi, Vmpp = Vm / 3574.0, pm = Po.p_V * 3574.0 * volume_const / Vm, Vr = (S.size_real.z + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * 0.25 * MC_pi,
        Vrpp = Vr / double(Pn), Vs = (S.L.x + 2.0 * Po.p_r) * (S.L.y + 2.0 * Po.p_r) * (S.L.z + 2.0 * Po.p_r);
    std::cerr << "Sample Vm=" << Vm << " Vmpp=" << Vmpp << " pm=" << pm << " Vr=" << Vr * volume_const << " Vrpp=" << Vrpp * volume_const << " " << Po.p_V * volume_const << " " << Vr / Vs << "\n";
    //std::cin.get();
    //std::cerr << "AAAA1\n";
    /*for (int k = 0; k < IL.N; ++k)
    {
        uint_fast32_t i = k/IL.IonP, j = IL.h_IL[k];
        float3 r;
        double rm;
        if (IL.h_ILtype[k] == 1)
        {
            r.x = P.h_R[j] - P.h_R[i];
            r.y = P.h_R[j + P.N] - P.h_R[i + P.N];
            r.z = P.h_R[j + 2 * P.N] - P.h_R[i + 2 * P.N];
            rm = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
            std::cerr << "I " << k << " " << i << " " << j << " " << IL.h_1d_iL[k] << " " << 1.0 / IL.h_1d_iL[k]
                << " | " << r.x << " " << r.y << " " << r.z << " " << rm << "\n";
        }
    }
    std::cin.get();/**/
    /*std::cerr << "Interaction list " << IL.N << "\n";
    cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < IL.N; ++i)
    {
        if (i % IL.IonP == 0)
            std::cerr << "\n" << i / IL.IonP;// << " (" << i % IL.IonP << ")";
        if (IL.h_IL[i] != UINT_FAST32_MAX)
            std::cerr << " " << IL.h_IL[i] << "&" << int(IL.h_ILtype[i]);
        else if (IL.h_IL[i] < UINT_FAST32_MAX && IL.h_IL[i] >= P.N)
            std::cerr << "!!!!";
        else if (IL.h_IL[i] == UINT_FAST32_MAX)        
            std::cerr << " M";       
    }
    std::cin.get();/**/
    //float3* tempIL_Rij = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Oijt = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Mijn = (float3*)malloc(IL.N * sizeof(float3));
    //float3* tempIL_Mijt = (float3*)malloc(IL.N * sizeof(float3));
    d_DeleteFarLinks << <A.bloks, SMEMDIM >> > (P.d_R, P.N, IL.d_IL, IL.d_ILtype, IL.d_1d_iL, IL.d_rij, IL.d_Oijt,
        IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, IL.IonP, Po.aa_farcut);
    //std::cin.get();
    std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point end;
    std::chrono::nanoseconds dr;

    uint_fast32_t nloaddelay = 0;
    R.stepsave = 0;
    //HANDLE_ERROR(cudaMemcpy(P.d_V, P.h_V, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(P.d_W, P.h_W, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    std::cerr << "Z " << Compress.Bottom << " " << Compress.Top << " " << A.ibloks * SMEMDIM << " " << IL.N << "\n";
    for (uint_fast32_t i = 0; i < 10000; ++i)//22.8374ms
    {
        //if(i%50==0)
        RenewInteractionList_BPM(P, C, S, A, IL);//H1.8991ms

        //RenewInteractionList_full(P, C, S, A, IL);
        if (true && i % 500 == 0)
        //if (i % 500 == 0)
        {
            //Po.m_E *= 2.0;
            HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_W, P.d_W, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
            sprintf(filename, "./result/steps/LAMMPS/CPc_%li.txt", i);
            SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
            std::cerr << "Save PBM " <<i<< "\n"; 
            std::cerr << "U " << Compress.center_d.z << " " << Compress.size_d.z << " " << 1e+3*(Compress.size_d.z - R.sz0)/ R.sz0
                << " | " << R.sFL[0] * force_const << " N " << R.sFL[1] * force_const << " N | "
                << R.sFL[0] * Compress._1d_Area * stress_const * 1e-6 << " MPa " << R.sFL[1] * Compress._1d_Area * stress_const * 1e-6 << " MPa | "
                << S.hidenpoint.x << " " << S.hidenpoint.y << " " << S.hidenpoint.z << "\n";
            //CheckDATA(P, Po, C, A, S, IL);
            /*HANDLE_ERROR(cudaMemcpy(P.h_F, P.d_F, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_M, P.d_M, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Oijt, IL.d_Oijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            for (int ii = 0; ii < P.N; ++ii)
                std::cerr << "P " << P.h_R[ii] - S.center.x << " " << P.h_R[ii + P.N] - S.center.y << " " << P.h_R[ii + 2 * P.N] - S.center.z << " | "
                << P.h_V[ii] << " " << P.h_V[ii + P.N] << " " << P.h_V[ii + 2 * P.N] << " | "
                << P.h_F[ii] << " " << P.h_F[ii + P.N] << " " << P.h_F[ii + 2 * P.N] << " | "
                << P.h_W[ii] << " " << P.h_W[ii + P.N] << " " << P.h_W[ii + 2 * P.N] << " | "
                << P.h_M[ii] << " " << P.h_M[ii + P.N] << " " << P.h_M[ii + 2 * P.N] << " | " << "\n";
            for (int ii = 0; ii < IL.N; ++ii)
                if (IL.h_IL[ii] != UINT_FAST32_MAX)
                    std::cerr << "I " << ii << " " << ii / IL.IonP << " " << IL.h_IL[ii] << " " << int(IL.h_ILtype[ii]) << " | "
                    << 1.0/IL.h_1d_iL[ii] << " " << IL.h_Oijt[ii].x << " " << IL.h_Oijt[ii].y << " " << IL.h_Oijt[ii].z << " | "
                    << "\n";
            std::cerr << Po.b_r * Po.b_r * MC_pi * Po.m_E * (P.h_R[5 + 2 * P.N] - P.h_R[4 + 2 * P.N] - 0.00201) * IL.h_1d_iL[150]
                << " " << P.h_R[5 + 2 * P.N] - P.h_R[4 + 2 * P.N] << "\n";
            std::cin.get();/**/
        }
        
        //d_CalculateForcesLJ << <A.bloks, SMEMDIM >> > (P.d_R, P.d_F, P.N, IL.d_IL, IL.IonP, 0.25f, 0.10f, float(1.0/0.10));
        /*HANDLE_ERROR(cudaMemcpy(tempIL_Rij, IL.d_rij, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(tempIL_Oijt, IL.d_Oijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(tempIL_Mijn, IL.d_Mijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(tempIL_Mijt, IL.d_Mijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));/**/

        //d_CheckBreakConditionDEM << <A.ibloks, SMEMDIM >> > (IL.d_IL, IL.d_1d_iL, IL.d_rij, IL.d_Oijt, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, 
        //    IL.d_ILtype, IL.N, Po.m_Ec, Po.m_Gc, Po.b_r);//H0.2104ms
        /*d_CalculateForcesDEM << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_W, IL.d_IL, IL.d_ILtype, IL.d_1d_iL,
            IL.d_rij, IL.d_Oijt, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, P.N, IL.IonP, Po.dt,
            Po.m_E, Po.m_G, Po.b_r,
            Po.m_E, Po.m_G, Po.p_A,
            Po.m_mu, Po.m_muroll, Po.m, Po.rP);/**/
        HANDLE_ERROR(cudaMemset((void*)P.d_F, 0, 3 * P.N * sizeof(float)));
        HANDLE_ERROR(cudaMemset((void*)P.d_M, 0, 3 * P.N * sizeof(float)));
        d_CalculateForcesDEM_22 << <4 * A.bloks, 256 >> > (P.d_R, P.d_V, P.d_W, IL.d_IL, IL.d_ILtype, IL.d_1d_iL,
            IL.d_rij, IL.d_Oijt, IL.d_Mijn, IL.d_Mijt, P.d_F, P.d_M, P.N, IL.IonP, Po.dt, Po.m_E, Po.m_G, Po.b_r, Po.m_Ec, Po.m_Gc);
        d_CalculateForcesDEM_21 << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_W, IL.d_IL, IL.d_ILtype, IL.d_1d_iL, IL.d_rij, IL.d_Oijt, P.d_F, P.d_M,
            P.N, IL.IonP, Po.dt, Po.hm_E, Po.hm_G, Po.p_A, Po.p_mu, Po.p_mur, Po.p_m, Po.p_r);
                
        
        

        //d_CalculateForcesDEM_1 << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_W, IL.d_IL, IL.d_ILtype, IL.d_1d_iL,
        //    IL.d_rij, IL.d_Oijt, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, P.N, IL.IonP, Po.dt,            
        //    Po.hm_E, Po.hm_G, Po.p_A, Po.p_mu, Po.p_mur, Po.p_m, Po.p_r);//H10.0058ms
        //d_CalculateForcesDEM_2 << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_W, IL.d_IL, IL.d_ILtype, IL.d_1d_iL,
        //    IL.d_rij, IL.d_Oijt, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, P.N, IL.IonP, Po.dt,
        //    Po.m_E, Po.m_G, Po.b_r);//H4.6395ms

        

        /*if (i % 1000 == 0)
        {
            HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Fijt, IL.d_Fijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Mijadd, IL.d_Mijadd, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            for (int kk = 0; kk < IL.N; ++kk)
            {
                if (kk % IL.IonP == 0)
                    std::cerr << "\n" << kk / IL.IonP;// << " (" << i % IL.IonP << ")";
                if (IL.h_IL[kk] != UINT_FAST32_MAX)
                    std::cerr << " " << IL.h_IL[kk] << "&" << int(IL.h_ILtype[kk]);
                else if (IL.h_IL[kk] < UINT_FAST32_MAX && IL.h_IL[kk] >= P.N)
                    std::cerr << "!!!!";
                else if (IL.h_IL[kk] == UINT_FAST32_MAX)
                    std::cerr << " M";
            }
            bool f1;
            for (int ii = 0; ii < P.N; ++ii)
            {
                for (int k1 = ii * IL.IonP; k1 < (ii + 1) * IL.IonP; ++k1)
                {
                    int jj = IL.h_IL[k1];
                    if ((jj < P.N) && (IL.h_ILtype[k1] == 1))
                    {
                        f1 = false;
                        double3 dM;
                        for (int k2 = jj * IL.IonP; k2 < (jj + 1) * IL.IonP; ++k2)
                            if (IL.h_IL[k2] == ii)
                            {
                                f1 = true;
                                dM.x = IL.h_Mijadd[k1].x - IL.h_Mijadd[k2].x;
                                dM.y = IL.h_Mijadd[k1].y - IL.h_Mijadd[k2].y;
                                dM.z = IL.h_Mijadd[k1].z - IL.h_Mijadd[k2].z;
                                if (fabs(dM.x) > 1e-10 || fabs(dM.y) > 1e-10 || fabs(dM.z) > 1e-10)
                                    std::cerr << "Error! Munsym " << ii << " " << jj << " | "
                                    << IL.h_Mijadd[k1].x << " " << IL.h_Mijadd[k1].y << " " << IL.h_Mijadd[k1].z << " | "
                                    << IL.h_Mijadd[k2].x << " " << IL.h_Mijadd[k2].y << " " << IL.h_Mijadd[k2].z << " | "
                                    << IL.h_Fijt[k1].x << " " << IL.h_Fijt[k1].y << " " << IL.h_Fijt[k1].z<< " | "
                                    << IL.h_Fijt[k2].x << " " << IL.h_Fijt[k2].y << " " << IL.h_Fijt[k2].z << "\n";
                            }
                        if (!f1)
                            std::cerr << "Error! Not find symmemtic link " << ii << " " << jj << "\n";
                    }
                }
            }
            //std::cin.get();
        }/**/

        //d_SumUpForcesDEM << <A.bloks, SMEMDIM >> > (IL.d_IL, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, 
        //    P.d_F, P.d_M, P.N, IL.IonP);//H6.1897ms
        //d_SumUpForcesDEMViscos << <A.bloks, SMEMDIM >> > (IL.d_IL, IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, P.d_F, P.d_M,
        //    P.d_V, P.d_W, P.N, IL.IonP, Po.nuV, Po.nuW);
        //std::cerr << "Arrays " << IL.d_Fijn << " " << IL.d_Fijt << " " << IL.d_Mijn << " " << IL.d_Mijt << " " << IL.d_Mijadd << " " << P.d_F << " " << P.d_M << " " << P.d_V << " " << P.d_W << " " << P.d_R << "\n"; std::cin.get();
        /*HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(P.h_W, P.d_W, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_b_r, IL.d_b_r, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_AxialMoment, IL.d_AxialMoment, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaMemcpy(IL.h_rij, IL.d_rij, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_Fijn, IL.d_Fijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_Fijt, IL.d_Fijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_Mijn, IL.d_Mijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_Mijt, IL.d_Mijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(IL.h_Mijadd, IL.d_Mijadd, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));

        h_CalculateForceElastic(P.h_R, P.h_V, P.h_W, P.h_F, P.h_M, IL.h_b_r, IL.h_1d_iL, IL.h_AxialMoment,
            tempIL_Rij, tempIL_Oijt, tempIL_Mijn, tempIL_Mijt, IL.h_Fijn, P.N, IL, P, Po.dt, 100.0f, 7.0f, 50.0f, 40.0f);/**/
        //HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
        //HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
        //for (int j = 0; j < IL.N; ++j)if (IL.h_ILtype[j] == 1)std::cerr << "ILtype " << j << " " << int(IL.h_ILtype[j]) << " " << j/IL.IonP << " " << IL.h_IL[j] << "\n";
        //d_CalculateForces << <A.bloks, SMEMDIM >> > (P.d_R, P.d_F, P.N, IL.d_IL, IL.IonP, Po.D);
        //d_CalculateIncrements << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.N, Po._1d_Mass_m_dt, Po.dt);
        //d_UniaxialCompression_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_F, R.d_FL, P.N, Compress.center, Compress.RR, Compress.Hd2, Compress.C, Compress.mu, S.center.z + S.spacesized2.z);
        //d_UniaxialCompression2_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_F, R.d_FL, P.N, Compress.center, Compress.RR, Compress.Hd2, Compress.C, Compress.mu, S.center.z + S.spacesized2.z);
        d_UniaxialCompression3_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_F, R.d_FL, P.N, Compress.center, Compress.RR, 
            Compress.Top, Compress.Bottom, Compress.TopR, Compress.BottomR, Compress.V, S.center.z + S.spacesized2.z);//H0.0011ms

        SumForcesULoading(R, Compress, i);//H0.4438ms
        d_CalculateIncrementsDEM << <A.bloks, SMEMDIM >> > (P.d_F, P.d_V, P.d_R, P.d_M, P.d_W, P.N, Po.dt_d_m, Po.dt, Po.dt_d_I);//H0.0533ms
        //std::cin.get();
        //d_CylinderRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.axis, S.center, S.R0, S.H0);
        //d_ParallelepipedRestriction << <A.bloks, SMEMDIM >> > (P.d_V, P.d_R, P.N, S.center, S.size);
        d_ParallelepipedCutRestriction << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.d_F, P.N, S.center, S.spacesized2, S.hidenpoint);//H0.055ms
        d_DeleteFarLinks << <A.bloks, SMEMDIM >> > (P.d_R, P.N, IL.d_IL, IL.d_ILtype, IL.d_1d_iL, IL.d_rij, IL.d_Oijt,
            IL.d_Fijn, IL.d_Fijt, IL.d_Mijn, IL.d_Mijt, IL.d_Mijadd, IL.IonP, Po.aa_farcut);//H0.3451ms
        //std::cin.get();
        if (fabs(R.sFL[0]) + fabs(R.sFL[1]) < 1e-8)
        {
            ++nloaddelay;
            Compress.V = -0.2*100 / velocity_const;
        }            
        else
        {
            nloaddelay = 0;
            Compress.V = -0.02*100 / velocity_const;
        }
        //if(i>10000)Compress.V = 0.0;
        //if (i > 20000 && nloaddelay > 5000)break;

        Compress.size_d.z += Compress.V * Po.dt;
        Compress.Top += Compress.V * Po.dt;
        Compress.TopR = Compress.Top - 2.0*Po.p_r;
        Compress.size.z = Compress.size_d.z;
        Compress.sized2.z = 0.5 * Compress.size_d.z;
        Compress.center_d.z += Compress.V * Po.dt;
        Compress.center.z = Compress.center_d.z;
        Compress.Hd2 = Compress.sized2.z;
        if (i % 1000 == 0)
        std::cerr << "U " << Compress.center_d.z << " " << Compress.size_d.z << " " << 1e+3 * (Compress.size_d.z - R.sz0) / R.sz0
            << " | " << R.sFL[0] * force_const << " N " << R.sFL[1] * force_const << " N | "
            << R.sFL[0] * Compress._1d_Area * stress_const * 1e-6 << " MPa " << R.sFL[1] * Compress._1d_Area * stress_const * 1e-6 << " MPa | "
            << S.hidenpoint.x << " " << S.hidenpoint.y << " " << S.hidenpoint.z << "\n";
        if (i % 100000 == 0)
            std::cerr << "UU " << Compress.V << " " << Po.dt << " " << Compress.size.z << " " << Compress.sized2.z << " " << Compress.Hd2 << "\n";
        //std::cerr << "FIn1\n";
        if (false && i % 10000 == 0 && i > 100000000)
        {
            HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_V, P.d_V, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_F, P.d_F, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_W, P.d_W, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(P.h_M, P.d_M, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Fijn, IL.d_Fijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Fijt, IL.d_Fijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Mijn, IL.d_Mijn, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Mijt, IL.d_Mijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Mijadd, IL.d_Mijadd, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_rij, IL.d_rij, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_Oijt, IL.d_Oijt, IL.N * sizeof(float3), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
            std::cerr << "Step " << i << "\n";
            for (int j = 0; j < P.N; ++j)
            {
                //std::cerr << j << " " << P.h_R[j] << " " << P.h_R[j + P.N] << " " << P.h_R[j + 2 * P.N] << " | " 
                //    << " " << P.h_F[j] << " " << P.h_F[j + P.N] << " " << P.h_F[j + 2 * P.N] << " || ";
                std::cerr << j << " " << P.h_R[j] << " " << P.h_R[j + P.N] << " " << P.h_R[j + 2 * P.N] << " | "
                    << normm(P.h_V, j, P.N) << " " << normm(P.h_F, j, P.N) << " " << normm(P.h_W, j, P.N) << " " << normm(P.h_M, j, P.N) << "\n";
            }/**/
            for (int j = 0; j < IL.N; ++j)
            {
                if (IL.h_IL[j] < P.N && IL.h_ILtype[j] == 1)
                    std::cerr << j << " " << int(IL.h_ILtype[j]) << " " << j / IL.IonP << " " << IL.h_IL[j]
                    << " | " << normm(IL.h_rij[j])-1.0/ IL.h_1d_iL[j] << " | " << normm(IL.h_Oijt[j])
                    << " | " << normm(IL.h_Fijn[j]) << " | " << normm(IL.h_Fijt[j])
                    << " | " << normm(IL.h_Mijn[j]) << " | " << normm(IL.h_Mijt[j]) << " | " << normm(IL.h_Mijadd[j])
                    << "\n";
                //std::cerr << j << " " << int(IL.h_ILtype[j]) << " " << j / IL.IonP << " " << IL.h_IL[j] 
                //    << "  | " << IL.h_rij[j].x << " " << IL.h_rij[j].y << " " << IL.h_rij[j].z
                //    << "  | " << IL.h_Oijt[j].x << " " << IL.h_Oijt[j].y << " " << IL.h_Oijt[j].z
                //    << "  | " << IL.h_Fijn[j].x << " " << IL.h_Fijn[j].y << " " << IL.h_Fijn[j].z
                //    << "  | " << IL.h_Fijt[j].x << " " << IL.h_Fijt[j].y << " " << IL.h_Fijt[j].z
                //    <<"\n";
            }/**/
            //SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
            //std::cin.get();
        }/**/
        //break;
        //std::cin.get();
    }
    end = std::chrono::high_resolution_clock::now();
    dr = end - begin;
    double calctime = std::chrono::duration_cast<std::chrono::milliseconds>(dr).count();
    std::cerr << "Fin time " << calctime << "ms " << calctime * 1e-3 / 60.0
        << "min " << calctime * 1e-4 << "ms" << " " << 22.2632 - calctime * 1e-4 << "ms "
        "faster " << 22.2632 / (calctime * 1e-4) << " \n";
    //std::cin.get();
    begin = end;
    sprintf(filename, "./result/Uniaxial.txt", 0);
    SaveSumForcesLoading(R, filename);
    std::cerr << "FIN CALCULATION!\n";
       //std::cin.get();
    /*RenewInteractionList_full(P, C, S, A, IL);
    std::cerr << "FIn2\n";
    HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", 2);
    SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);
    //std::cin.get();
    RenewInteractionList_full(P, C, S, A, IL);
    std::cerr << "FIn3\n";
    HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    sprintf(filename, "./result/steps/LAMMPS/CP_%li.txt", 3);
    SaveLammpsDATASimple(P, C, S, A, IL, Po, filename);/**/
    /*CellDistributionInit(P, A, S, C);
    CellDistribution(P, A, S, C);
    InteractionListInit(P, A, S, IL);
    InteractionListConstruct(P, A, S, IL, C);*/
    std::cin.get();
    // Add vectors in parallel.
    cudaError_t cudaStatus ;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    deleteArrays(A, P, C, IL, R);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

