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
#include <chrono>
//#define defc_SaveLammps

void cuc_freePD(md_task_data& mdTD)
{
    if (mdTD.P.d_R != nullptr) { cudaFree(mdTD.P.d_R); mdTD.P.d_R = nullptr; }
    if (mdTD.P.d_F != nullptr) { cudaFree(mdTD.P.d_F); mdTD.P.d_F = nullptr; }
    if (mdTD.P.d_V != nullptr) { cudaFree(mdTD.P.d_V); mdTD.P.d_V = nullptr; }
    if (mdTD.P.d_M != nullptr) { cudaFree(mdTD.P.d_M); mdTD.P.d_M = nullptr; }
    if (mdTD.P.d_W != nullptr) { cudaFree(mdTD.P.d_W); mdTD.P.d_W = nullptr; }
}

void cuc_createPD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_V, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_F, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_M, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_W, 3 * mdTD.P.N * sizeof(float)));
    //HANDLE_ERROR(cudaMalloc((void**)&mdTD.P.d_Q, P.N * sizeof(float4)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_V, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_F, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_W, 0, 3 * mdTD.P.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_M, 0, 3 * mdTD.P.N * sizeof(float)));
    //HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_Q, 0, P.N * sizeof(float4)));
#ifdef defc_SaveLammps
    mdTD.P.h_R = (float*)malloc(3 * mdTD.P.N * sizeof(float));
#endif // defc_SaveLammps

    
}

void cuc_freeCD(md_task_data& mdTD)
{
    if (mdTD.C.d_tmp_old != nullptr) { cudaFree(mdTD.C.d_tmp_old); mdTD.C.d_tmp_old = nullptr; mdTD.C.d_tmp = nullptr; }
    if (mdTD.C.d_IP != nullptr) { cudaFree(mdTD.C.d_IP); mdTD.C.d_IP = nullptr; }
    if (mdTD.C.d_CI != nullptr) { cudaFree(mdTD.C.d_CI); mdTD.C.d_CI = nullptr; }
    if (mdTD.C.d_CIs != nullptr) { cudaFree(mdTD.C.d_CIs); mdTD.C.d_CIs = nullptr; }
    if (mdTD.C.d_pnC != nullptr) { cudaFree(mdTD.C.d_pnC); mdTD.C.d_pnC = nullptr; }
}

void cuc_createCD(md_task_data& mdTD)
{
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_IP, mdTD.P.N * sizeof(uint_fast32_t)));
    d_FillIndex <<< mdTD.A.bloks, SMEMDIM >> > (mdTD.C.d_IP, mdTD.P.N);
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CI, mdTD.P.N * sizeof(uint_fast32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_CIs, 2 * mdTD.P.N * sizeof(uint_fast32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_pnC, 2 * mdTD.C.N * sizeof(uint_fast32_t)));
    mdTD.C.dtmpN_old = 128;
    HANDLE_ERROR(cudaMalloc((void**)&mdTD.C.d_tmp_old, mdTD.C.dtmpN_old));
}

void cuc_freeID(md_task_data& mdTD)
{
    //std::cerr << "fid " << mdTD.IL.d_IL << " " << mdTD.IL.d_ILtype << " " << mdTD.IL.d_1d_iL << "\n";
    if (mdTD.IL.d_IL != nullptr) { cudaFree(mdTD.IL.d_IL); mdTD.IL.d_IL = nullptr; }
    if (mdTD.IL.d_ILtype != nullptr) { cudaFree(mdTD.IL.d_ILtype); mdTD.IL.d_ILtype = nullptr; }
    if (mdTD.IL.d_1d_iL != nullptr) { cudaFree(mdTD.IL.d_1d_iL); mdTD.IL.d_1d_iL = nullptr; }
    if (mdTD.IL.d_rij != nullptr) { cudaFree(mdTD.IL.d_rij); mdTD.IL.d_rij = nullptr; }
    if (mdTD.IL.d_Oijt != nullptr) { cudaFree(mdTD.IL.d_Oijt); mdTD.IL.d_Oijt = nullptr; }
    if (mdTD.IL.d_Fijn != nullptr) { cudaFree(mdTD.IL.d_Fijn); mdTD.IL.d_Fijn = nullptr; }
    if (mdTD.IL.d_Fijt != nullptr) { cudaFree(mdTD.IL.d_Fijt); mdTD.IL.d_Fijt = nullptr; }
    if (mdTD.IL.d_Mijn != nullptr) { cudaFree(mdTD.IL.d_Mijn); mdTD.IL.d_Mijn = nullptr; }
    if (mdTD.IL.d_Mijt != nullptr) { cudaFree(mdTD.IL.d_Mijt); mdTD.IL.d_Mijt = nullptr; }
    if (mdTD.IL.d_Mijadd != nullptr) { cudaFree(mdTD.IL.d_Mijadd); mdTD.IL.d_Mijadd = nullptr; }
    
}

void cuc_createID(md_task_data& mdTD)
{
    //uint_fast64_t stmp = IL.N * sizeof(uint_fast32_t) + IL.N * sizeof(uint_fast8_t) + IL.N * sizeof(float) + 7 * IL.N * sizeof(float3);
    //std::cerr << "InteractionListInit " << IL.N << " " << stmp << " " << stmp / (1024 * 1024) << " | " << sqrt(IL.aacut) << "\n";
    //std::cin.get();
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_IL), mdTD.IL.N * sizeof(uint_fast32_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_ILtype), mdTD.IL.N * sizeof(uint_fast8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_1d_iL), mdTD.IL.N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_rij), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Oijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Fijn), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Fijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijn), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijt), mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMalloc((void**)&(mdTD.IL.d_Mijadd), mdTD.IL.N * sizeof(float3)));

    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_ILtype, 0, mdTD.IL.N * sizeof(uint_fast8_t)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_1d_iL, 0, mdTD.IL.N * sizeof(float)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_rij, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Oijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Fijn, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Fijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijn, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijt, 0, mdTD.IL.N * sizeof(float3)));
    HANDLE_ERROR(cudaMemset((void*)mdTD.IL.d_Mijadd, 0, mdTD.IL.N * sizeof(float3)));
#ifdef defc_SaveLammps
    mdTD.IL.h_IL = (uint_fast32_t*)malloc(mdTD.IL.N * sizeof(uint_fast32_t));
    mdTD.IL.h_ILtype = (uint_fast8_t*)malloc(mdTD.IL.N * sizeof(uint_fast8_t));
#endif // defc_SaveLammps

    
    //mdTD.IL.h_1d_iL = (float*)malloc(mdTD.IL.N * sizeof(float));
}

void h_CalculationUniaxialCompression(md_task_data& mdTD, float *h_R)
{
    char filename[256] = "";
    uint_fast32_t steps, i;
    mdTD.IL.CalculateParameters(mdTD.P.N, mdTD.Po.a_farcut);
    mdTD.C.CalculateParameters(mdTD.S.B.x, mdTD.S.B.y, mdTD.S.B.z);
    mdTD.A.ibloks = ceil(mdTD.IL.N / (SMEMDIM)) + 1;
    mdTD.A.CalculateParameters(mdTD.P.N, mdTD.IL.N);
    cuc_freePD(mdTD); //std::cerr << "q1\n";
    cuc_createPD(mdTD); //std::cerr << "q2\n";
    cuc_freeCD(mdTD); //std::cerr << "q3\n";
    cuc_createCD(mdTD); //std::cerr << "q4\n";
    cuc_freeID(mdTD); //std::cerr << "q5\n";
    cuc_createID(mdTD); //std::cerr << "D1\n";//std::cerr << "q6\n";
    mdTD.R.Nsave = 1000000; mdTD.R.dNsave = 1;
    ResultsUInit(mdTD.P, mdTD.A, mdTD.S, mdTD.C, mdTD.R, mdTD.Compress); //std::cerr << "D2\n"; std::cerr << mdTD.P.d_R<<" "<<h_R<<" "<< mdTD.P.N <<"\n";
    HANDLE_ERROR(cudaMemcpy(mdTD.P.d_R, h_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyHostToDevice)); std::cerr << "D3\n";
    std::cerr << "Material " << mdTD.Po.m_E << " " << mdTD.Po.m_Ec << " " << mdTD.Po.m_Gc << " " << mdTD.Po.b_r << " " << mdTD.Po.b_d << " | " << mdTD.Po.hm_E << " " << mdTD.Po.b_dd << "\n";
    CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C); //std::cerr << "D4\n";
    InteractionListConstruct(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C); //std::cerr << mdTD.IL.N << " " << mdTD.IL.IonP << "\n"; std::cerr << "D5\n";
    d_ConstructBoundInteractions << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W,
        mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.P.N, mdTD.IL.IonP, mdTD.Po.b_dd, mdTD.Po.b_r); //std::cerr << mdTD.IL.N<<" "<< mdTD.Po.b_dd << "\n"; std::cerr << "D6\n";
    //HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
    //for (i = 0; i < mdTD.IL.N; ++i)
    //{
    //    if (i % 32 == 0)std::cerr << "\n"<<i/32<<" ";
    //    std::cerr << i << " " << mdTD.IL.h_IL << "(" << mdTD.IL.h_ILtype << ") ";
    //}
    d_DeleteFarLinks << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt,
        mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.IL.IonP, mdTD.Po.aa_farcut); //std::cerr << "D7\n";

    std::chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point end;
    std::chrono::nanoseconds dr;
    //std::cin.get();
    uint_fast32_t nloaddelay = 0;
    mdTD.R.stepsave = 0;
    //HANDLE_ERROR(cudaMemcpy(P.d_V, P.h_V, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(P.d_W, P.h_W, 3 * P.N * sizeof(float), cudaMemcpyHostToDevice));
    //std::cerr << "Z " << Compress.Zb << " " << Compress.Zt << " " << A.ibloks * SMEMDIM << " " << IL.N << "\n";
    //for (uint_fast32_t i = 0; i < 10000 && !mdTD.Compress.Fracture; ++i)//22.8374ms
    
    for (steps = 0; mdTD.Compress.Strain < mdTD.Compress.MaxStrain && !mdTD.Compress.Fracture; ++steps)
    {
#ifdef defc_SaveLammps
        if (steps % 1000 == 0)
        {
            HANDLE_ERROR(cudaMemcpy(mdTD.P.h_R, mdTD.P.d_R, 3 * mdTD.P.N * sizeof(float), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_IL, mdTD.IL.d_IL, mdTD.IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(mdTD.IL.h_ILtype, mdTD.IL.d_ILtype, mdTD.IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
            sprintf(filename, "./result/steps/LAMMPS/CPu_%li.txt", steps);
            SaveLammpsDATASimple(mdTD.P, mdTD.C, mdTD.S, mdTD.A, mdTD.IL, mdTD.Po, filename, false);
            std::cerr << "Stress " << steps << " " << mdTD.Compress.Stress * stress_const * 1e-6 << " " << mdTD.Compress.Stress << "\n";
        }
#endif // defc_SaveLammps        
        CellDistribution(mdTD.P, mdTD.A, mdTD.S, mdTD.C);// std::cerr << "D8\n";
        InteractionListReConstructBPM(mdTD.P, mdTD.A, mdTD.S, mdTD.IL, mdTD.C);// std::cerr << "D9\n";
        HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_F, 0, 3 * mdTD.P.N * sizeof(float)));
        HANDLE_ERROR(cudaMemset((void*)mdTD.P.d_M, 0, 3 * mdTD.P.N * sizeof(float)));
        d_CalculateForcesDEM_22 << <4 * mdTD.A.bloks, 256 >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL,
            mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.P.d_F, mdTD.P.d_M, mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.m_E, mdTD.Po.m_G, mdTD.Po.b_r, mdTD.Po.m_Ec, mdTD.Po.m_Gc);// std::cerr << "D10\n";
        d_CalculateForcesDEM_21 << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_W, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt, mdTD.P.d_F, mdTD.P.d_M,
            mdTD.P.N, mdTD.IL.IonP, mdTD.Po.dt, mdTD.Po.hm_E, mdTD.Po.hm_G, mdTD.Po.p_A, mdTD.Po.p_mu, mdTD.Po.p_mur, mdTD.Po.p_m, mdTD.Po.p_r);// std::cerr << "D11\n";

               
        d_UniaxialCompression3_simple << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.R.d_FL, mdTD.P.N, mdTD.Compress.center, mdTD.Compress.RR,
            mdTD.Compress.Top, mdTD.Compress.Bottom, mdTD.Compress.TopR, mdTD.Compress.BottomR, mdTD.Compress.V, mdTD.S.center.z + mdTD.S.spacesized2.z);// std::cerr << "D12\n";//H0.0011ms

        SumForcesULoading(mdTD.R, mdTD.Compress, steps);// std::cerr << "D12.5\n";//H0.4438ms
        d_CalculateIncrementsDEM << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_F, mdTD.P.d_V, mdTD.P.d_R, mdTD.P.d_M, mdTD.P.d_W, mdTD.P.N, mdTD.Po.dt_d_m, mdTD.Po.dt, mdTD.Po.dt_d_I);// std::cerr << "D13\n";//H0.0533ms
        
        d_ParallelepipedCutRestriction << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.d_V, mdTD.P.d_F, mdTD.P.N, mdTD.S.center, mdTD.S.spacesized2, mdTD.S.hidenpoint);// std::cerr << "D14\n";//H0.055ms
        d_DeleteFarLinks << <mdTD.A.bloks, SMEMDIM >> > (mdTD.P.d_R, mdTD.P.N, mdTD.IL.d_IL, mdTD.IL.d_ILtype, mdTD.IL.d_1d_iL, mdTD.IL.d_rij, mdTD.IL.d_Oijt,
            mdTD.IL.d_Fijn, mdTD.IL.d_Fijt, mdTD.IL.d_Mijn, mdTD.IL.d_Mijt, mdTD.IL.d_Mijadd, mdTD.IL.IonP, mdTD.Po.aa_farcut);// std::cerr << "D15\n";//H0.3451ms
        
        mdTD.Compress.CompressU(mdTD.Compress.Stress, mdTD.Po);
        //if (i % 1000 == 0)
        //    std::cerr << "U " << Compress.center_d.z << " " << Compress.size_d.z << " " << 1e+3 * (Compress.size_d.z - R.sz0) / R.sz0
        //    << " | " << R.sFL[0] * force_const << " N " << R.sFL[1] * force_const << " N | "
        //    << R.sFL[0] * Compress._1d_Area * stress_const * 1e-6 << " MPa " << R.sFL[1] * Compress._1d_Area * stress_const * 1e-6 << " MPa | "
        //    << S.hidenpoint.x << " " << S.hidenpoint.y << " " << S.hidenpoint.z << "\n";
        //if (i % 100000 == 0) std::cerr << "UU " << Compress.V << " " << Po.dt << " " << Compress.size.z << " " << Compress.sized2.z << " " << Compress.Hd2 << "\n";
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
    SaveSumForcesLoading(mdTD.R, filename);
    std::cerr << "FIN CALCULATION!\n";

    cuc_freePD(mdTD); //std::cerr << "q1\n";
    cuc_freeCD(mdTD); //std::cerr << "q3\n";
    cuc_freeID(mdTD); //std::cerr << "q5\n";
    ResultsUDelete(mdTD.R);
    //R.Nsave = 1000000;
    //R.dNsave = 1;
    

    

    //std::cerr << "T " << P.N << " " << IL.N << " " << IL.IonP << " " << A.bloks << " " << SMEMDIM << "\n";    
    //std::cerr << "Cut " << Compress.Zt << " " << Compress.Zb << " | " << S.B_real.z << " " << S.A_real.z << " " << Po.p_r << " | " <<Compress.Area*length_const*length_const << "\n";
    //d_CutCylinderSpecimen_simple << <A.bloks, SMEMDIM >> > (P.d_R, P.d_V, P.N, Compress.center, Compress.sized2.x* Compress.sized2.x, Compress.Hd2, S.hidenpoint);
    //std::cin.get();
    //CellDistributionInit(P, A, S, C);
    //std::cerr << "Bound " << Po.b_d << " " << Po.b_dd << " " << sqrt(Po.b_dd) << "\n";       
    //std::cin.get();
    //std::cerr << "AAAA\n";
    //HANDLE_ERROR(cudaMemcpy(IL.h_IL, IL.d_IL, IL.N * sizeof(uint_fast32_t), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(P.h_R, P.d_R, 3 * P.N * sizeof(float), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(IL.h_1d_iL, IL.d_1d_iL, IL.N * sizeof(float), cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(IL.h_ILtype, IL.d_ILtype, IL.N * sizeof(uint_fast8_t), cudaMemcpyDeviceToHost));
    /*uint_fast32_t i, j, ILn = 0, ILn0 = 0, Pn = 0;
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
    
    std::cerr << "Interactions Number " << ILn << " " << ILn0 << " " << Pn << "\n";/**/
    //std::cin.get();
    //double Vm = 0.05 * 0.0125 * 0.0125 * MC_pi, Vmpp = Vm / 3574.0, pm = Po.p_V * 3574.0 * volume_const / Vm, Vr = (S.size_real.z + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * (S.size_real.x + 2.0 * Po.p_r) * 0.25 * MC_pi,
    //    Vrpp = Vr / double(Pn), Vs = (S.L.x + 2.0 * Po.p_r) * (S.L.y + 2.0 * Po.p_r) * (S.L.z + 2.0 * Po.p_r);
    //std::cerr << "Sample Vm=" << Vm << " Vmpp=" << Vmpp << " pm=" << pm << " Vr=" << Vr * volume_const << " Vrpp=" << Vrpp * volume_const << " " << Po.p_V * volume_const << " " << Vr / Vs << "\n";
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
    
    //std::cin.get();
    
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
    //std::cin.get();
    //cudaError_t cudaStatus;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    

    
}

