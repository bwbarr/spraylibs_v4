import numpy as np
from scipy.optimize import fsolve
from .ssgf import sourceM_F09_prev,sourceM_F09_dev1,sourceM_F94
from .util import fall_velocity_PK97,qsat0,satratio,stabIntH,stabIntM,stabIntSprayH,\
        thermo_HRspr,thermo_HTspr,tq_zR_fsolve_residual

def spraymedHF(z_1,U_1,th_1,q_1,p_0,fs,eps,dcp,swh,mss,ustar,t_0,r0=None,delta_r0=None,SSGFname='F09_dev1',feedback=True,getprofiles=False,zRvaries=False,stability=True,sprayLB=10.0,fdbkfsolve='fsolve',fdbkcrzyOPT=0,showfdbkcrzy=False,scaleSSGF=False,chi1=None,chi2=None,which_z0tq='Garratt1994'):
    """
    Model for spray-mediated heat fluxes.
    Parameters:
        z_1 - height of lowest model level [m]
        U_1 - windspeed magnitude at lowest model level [m s-1]
        th_1 - potential temperature at lowest model level [K]
        q_1 - specific humidity at lowest model level [kg kg-1]
        p_0 - surface pressure [Pa]
        fs - scale factor on droplet SSGF (this is Chris' sourcestrength) [-]
        eps - wave energy dissipation flux [kg s-3]
        dcp - dominant phase speed [m s-1]
        swh - significant wave height [m]
        mss - mean squared waveslope [-]
        ustar - friction velocity [m s-1]
        t_0 - sea surface temperature [K]
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width [m]
        SSGFname - name of SSGF to use [-]
        feedback - True to model feedback
        getprofiles - True to generate theta and q vertical profiles within surface layer
        zRvaries - True to iteratively determine zR
        stability - True to include stability
        sprayLB - Lower bound on U_10 for calculating spray heat fluxes [m s-1]
        fdbkfsolve - 'fsolve' to solve feedback using fsolve; 'iterNoIG' for set number of iterations
                with no IG; 'iterIG' for set number of iterations with IG
        fdbkcrzyOPT - Option for dealing with poorly behaved feedback points
        showfdbkcrzy - True to show some value where feedback is poorly behaved, False to mask with nans
        scaleSSGF - True to scale SSGF to favor large or small droplets using chi1 and chi2
        chi1 - factor scaling small droplet end of SSGF
        chi2 - factor scaling large droplet end of SSGF
        which_z0tq - 'Garratt1994' to calculate z0t and z0q based on Garratt (1994), 
                'FairallEA2003' to use Fairall et al. (2003) i.e. COARE 3.0
    """
    # 1. Constants and non-varying parameters =========================================================
    kappa = 0.41    # von Karman constant [-]
    g = 9.81    # Acceleration due to gravity [m s-2]
    Rdry = 287.    # Dry air gas constant [J kg-1 K-1]
    rho_sw = 1030.    # Density of seawater [kg m-3]
    rho_dry = 2160.    # Density of chrystalline salt [kg m-3]
    cp_sw = 4200.    # Specific heat capacity of seawater [J kg-1 K-1]
    cp_a = 1004.67    # Specific heat capacity of air [J kg-1 K-1]
    Lv = 2.43e6    # Latent heat of vap for water at 30C [J kg-1] (EngineeringToolbox.com)
    Pr_a = 0.71    # Prandtl number for air [-]
    Sc_a = 0.60    # Schmidt number for air [-]
    nu = 2    # Number of ions into which NaCl dissociates [-]
    Phi_s = 0.924    # Practical osmotic coefficient at molality of 0.6 [-]
    Mw = 18.02    # Molecular weight of water [g mol-1]
    Ms = 58.44    # Molecular weight of salt [g mol-1]
    xs = np.full_like(z_1,0.035)    # Mass fraction of salt in seawater [-]
    th2t = (p_0/1e5)**0.286    # Factor converting potential temperature to temperature [-]
    rdryBYr0 = (xs*rho_sw/rho_dry)**(1/3)    # Ratio of rdry to r0 [-]
    y0 = -nu*Phi_s*Mw/Ms*rho_dry/rho_sw*rdryBYr0**3/(1 - rho_dry/rho_sw*rdryBYr0**3)    # y for surface seawater [-]
    q_0 = qsat0(t_0,p_0)*(1 + y0)    # Specific humidity at surface (accounting for salt) [kg kg-1]
    t_1 = th_1*th2t    # Temperature at z_1 [K]
    t_mean = 0.5*(t_1+t_0)    # Approx mean air temperature [K], for calculating properties
    tC_mean = t_mean - 273.15    # Approx mean air temperature [C], for calculating properties
    q_mean = 0.5*(q_1+q_0)    # Approx mean air spec hum [kg kg-1], for calculating properties
    rho_a = p_0/(Rdry*t_mean*(1.+0.61*q_mean))    # Air density [kg m-3]
    k_a = 2.411e-2*(1.+3.309e-3*tC_mean-1.441e-6*tC_mean**2)    # Thermal conductivity of air [W m-1 K-1]
    nu_a = 1.326e-5*(1.+6.542e-3*tC_mean+8.301e-6*tC_mean**2-4.84e-9*tC_mean**3)    # Kin visc of air [m2 s-1]
    Dv_a = 2.11e-5*((tC_mean+273.)/273.)**1.94    # Water vapor diffusivity of air [m2 s-1]
    gammaWB = 240.97*17.502/(tC_mean+240.97)**2    # gamma = (dqsat/dT)/qsat [K-1], per Buck (1981) correlation
    G_S = rho_a*cp_a*kappa*ustar    # Dimensional group for SHF [W m-2 K-1]
    G_L = rho_a*Lv*kappa*ustar    # Dimensional group for LHF [W m-2]
    delspr = np.minimum(swh,z_1)    # Spray layer thickness [m], nominally one swh per M&V2014a, limited to z_1
    th_0 = t_0/th2t    # Surface potential temperature [K]
    H_tol = 0.01    # Tolerance on convergence for heat fluxes [W m-2]
    
    # 2. Interfacial heat fluxes and related parameters ================================================
    L = np.where(np.isnan(ustar),np.nan,-1e12)    # Obukhov stability length [m]
    if stability:    # Iterate to determine L
        z0      = np.full_like(z_1,np.nan)    # Momentum roughness length [m]
        z0t     = np.full_like(z_1,np.nan)    # Thermal roughness length [m]
        z0q     = np.full_like(z_1,np.nan)    # Moisture roughness length [m]
        psiH_1  = np.full_like(z_1,np.nan)    # Stability integral for heat at z_1 [-]
        H_S0pr  = np.full_like(z_1,np.nan)    # Sensible heat flux without spray [W m-2]
        H_L0pr  = np.full_like(z_1,np.nan)    # Latent heat flux without spray [W m-2]
        NC = ~np.isnan(ustar)    # Non-converged gridpoints
        firstCalc = True
        count = 0
        while np.nansum(NC) > 0:
            count += 1
            if count > 100:
                L[NC] = -1e12    # Where iteration did not converge, revert to no stability
                break
            print('Interfacial HF iteration %d: %d non-converged points' % (count,np.nansum(NC)))
            H_S0pr_prev = np.copy(H_S0pr[NC])    # Non-converged values from previous iteration [W m-2]
            H_L0pr_prev = np.copy(H_L0pr[NC])    # Non-converged values from previous iteration [W m-2]
            thvstar = -H_S0pr[NC]*kappa/G_S[NC] - 0.61*th_1[NC]*H_L0pr[NC]*kappa/G_L[NC]    # Flux scale for thv [K]
            if firstCalc:
                firstCalc = False
            else:
                L[NC] = ustar[NC]**2/(kappa*g/th_1[NC]*thvstar)
            psiM_1 = stabIntM(z_1[NC]/L[NC])    # Stability integral for momentum at z_1 [-]
            z0[NC] = z_1[NC]/np.exp(kappa*U_1[NC]/ustar[NC] + psiM_1)
            Restar = ustar[NC]*z0[NC]/nu_a[NC]    # Roughness Reynolds number [-]
            if which_z0tq == 'Garratt1994':
                z0t[NC] = z0[NC]/np.exp(kappa*(7.3*Restar**0.25*Pr_a**0.5 - 5))    # Per Garratt (1992) Eq. 4.14
                z0q[NC] = z0[NC]/np.exp(kappa*(7.3*Restar**0.25*Sc_a**0.5 - 5))    # Per Garratt (1992) Eq. 4.15
            elif which_z0tq == 'FairallEA2003':
                z0t[NC] = np.minimum(1.1e-4,5.5e-5*Restar**-0.6)    # COARE 3.0, Fairall et al. (2003) Eq. 28
                z0q[NC] = np.minimum(1.1e-4,5.5e-5*Restar**-0.6)    # COARE 3.0, Fairall et al. (2003) Eq. 28
            psiH_1[NC] = stabIntH(z_1[NC]/L[NC])
            H_S0pr[NC] = G_S[NC]*(th_0[NC] - th_1[NC])/(np.log(z_1[NC]/z0t[NC]) - psiH_1[NC])
            H_L0pr[NC] = G_L[NC]*( q_0[NC] -  q_1[NC])/(np.log(z_1[NC]/z0q[NC]) - psiH_1[NC])
            NC[NC] = np.where(np.logical_and(np.abs(H_S0pr[NC] - H_S0pr_prev) < H_tol,\
                                             np.abs(H_L0pr[NC] - H_L0pr_prev) < H_tol),False,True)
        L[np.abs(L) == np.inf] = -1e12    # Where L went to +/- infinity, revert to no stability
    # Run one more time (if stability) or for the first time (no stability) with the final values of L
    psiM_1 = stabIntM(z_1/L)
    z0 = z_1/np.exp(kappa*U_1/ustar + psiM_1)
    Restar = ustar*z0/nu_a
    if which_z0tq == 'Garratt1994':
        z0t = z0/np.exp(kappa*(7.3*Restar**0.25*Pr_a**0.5 - 5))
        z0q = z0/np.exp(kappa*(7.3*Restar**0.25*Sc_a**0.5 - 5))
    elif which_z0tq == 'FairallEA2003':
        z0t = np.minimum(1.1e-4,5.5e-5*Restar**-0.6)
        z0q = np.minimum(1.1e-4,5.5e-5*Restar**-0.6)
    psiH_1 = stabIntH(z_1/L)
    H_S0pr = G_S*(th_0 - th_1)/(np.log(z_1/z0t) - psiH_1)
    H_L0pr = G_L*( q_0 -  q_1)/(np.log(z_1/z0q) - psiH_1)
    # Calculate relevant diagnosed parameters
    psiH_delspr   = stabIntH(delspr/L)    # Stability integral for heat at delspr [-]
    psiH_delsprD2 = stabIntH(delspr/2/L)    # Stability integral for heat at delspr/2 [-]
    phisprH_delspr   = stabIntSprayH(delspr/L)    # Stability integral for heat with spray at delspr [-]
    phisprH_delsprD2 = stabIntSprayH(delspr/2/L)    # Stability integral for heat with spray at delspr/2 [-]
    t_delsprD2pr = (th_0 - H_S0pr/G_S*(np.log(delspr/2/z0t) - psiH_delsprD2))*th2t    # t at mid-layer w/o fdbk [K]
    q_delsprD2pr =   q_0 - H_L0pr/G_L*(np.log(delspr/2/z0q) - psiH_delsprD2)    # q at mid-layer w/o fdbk [kg kg-1]
    s_delsprD2pr = satratio(t_delsprD2pr,p_0,q_delsprD2pr,0.99999)    # s at mid-layer w/o fdbk [-]
    if feedback:    # Interfacial feedback coefficients for SHF and LHF [-]
        gamma_S = (np.log(delspr/z0t) - psiH_delspr - 1 + phisprH_delspr)/(np.log(z_1/z0t) - psiH_1)
        gamma_L = (np.log(delspr/z0q) - psiH_delspr - 1 + phisprH_delspr)/(np.log(z_1/z0q) - psiH_1)
    else:
        gamma_S = np.full_like(z_1,1.0)
        gamma_L = np.full_like(z_1,1.0)

    # 3. Background calculations for spray heat fluxes =====================================================
    # Specific available energy
    t_10Npr = (th_0 - H_S0pr/G_S*np.log(10/z0t))*th2t    # 10m neutral temp without spray [K]
    q_10Npr =   q_0 - H_L0pr/G_L*np.log(10/z0q)    # 10m neutral spec hum without spray [kg kg-1]
    s_10Npr = satratio(t_10Npr,p_0,q_10Npr,0.99999)    # 10m neutral saturation ratio without spray [-]
    betaWB_10Npr = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0(t_10Npr,p_0))    # 10m neutral WB coeff without spray [-]
    wetdep_10Npr = (1 - s_10Npr/(1 + y0))*(1 - betaWB_10Npr)/gammaWB    # 10m neutral WB depression without spray [K]
    reqBYr0_10Npr = (xs*(1 + nu*Phi_s*Mw/Ms/(1 - s_10Npr)))**(1/3)    # 10m neutral req/r0 without spray [-]
    a_T = cp_sw*(t_0 - t_10Npr + wetdep_10Npr)    # Available energy for heat transfer due to temp change [J kg-1]
    a_R = Lv*(1 - reqBYr0_10Npr**3)    # Available energy for heat transfer due to size change [J kg-1]
    # SSGF and droplet hydrodynamics
    U_10  = ustar/kappa*(np.log(10/z0) - stabIntM(10/L))    # 10m windspeed [m s-1]
    U_10N = ustar/kappa*(np.log(10/z0))    # 10m neutral windspeed [m s-1]
    if SSGFname == 'F09_prev':
        C5 = 0.04    # Chris' coefficient [-]
        C6 = 20    # Chris' coefficient [-]
        h_gust = np.maximum(C5*swh/2,C6*z0)    # Gust height [m]
        U_h = ustar/kappa*(np.log(h_gust/z0) - stabIntM(h_gust/L))    # Windspeed at gust height [m s-1]
        r0,delta_r0,M_spr,dmdr0 = sourceM_F09_prev(fs,eps,swh,dcp,mss,U_10,U_h,r0=r0,delta_r0=delta_r0)    # [m],[m],[kg m-2 s-1],[kg m-2 s-1 m-1]
    elif SSGFname == 'F09_dev1':
        model_coeffs = [fs,1.35,0.1116,0.719,2.17,0.852]    # [sourcestrength,mag,exp,mss,sigma_h,erf] Mic tuning
        r0,delta_r0,M_spr,dmdr0 = sourceM_F09_dev1(model_coeffs,eps,swh,dcp,mss,U_10,ustar,z0,L,r0=r0,delta_r0=delta_r0)    # [m],[m],[kg m-2 s-1],[kg m-2 s-1 m-1]
        if scaleSSGF:    # Scale SSGF to favor large or small droplets
            r0,delta_r0,M_spr_scale,dmdr0_scale = sourceM_F09_dev1(model_coeffs,eps,swh,dcp,mss,U_10,ustar,z0,L,r0=r0,delta_r0=delta_r0,chi1=chi1,chi2=chi2)
            fMspr = M_spr/M_spr_scale
            dmdr0 = dmdr0_scale*fMspr
    elif SSGFname in ['F94_MOM80','F94_CF21']:
        if SSGFname == 'F94_MOM80':
            Wform = 'MOM80'
        elif SSGFname == 'F94_CF21':
            Wform = 'CF21'
        r0,delta_r0,M_spr,dmdr0 = sourceM_F94(fs,U_10,Wform,r0=r0,delta_r0=delta_r0)    # [m],[m],[kg m-2 s-1],[kg m-2 s-1 m-1]
    r0_rng = np.arange(np.size(r0))    # List to use for stepping through r0
    v_g = fall_velocity_PK97(r0)    # Droplet settling velocity [m s-1]
    tauf = np.array([delspr/v_g[i] for i in r0_rng])    # Characteristic droplet settling time [s]
    Fp = np.array([1.+0.25*(2.*v_g[i]*r0[i]/nu_a)**0.5 for i in r0_rng])    # Slip factor (Pr&Kl) [-]
    nospray = np.isnan(eps)
    zerospray = np.logical_and(~nospray,U_10 < sprayLB)    # Eliminate points below lower bound
    # Heat flux due to temp change
    tauT = np.array([np.where(np.logical_or(nospray,zerospray),np.nan,\
            rho_sw*cp_sw*r0[i]**2/3./k_a/Fp[i,:,:]) for i in r0_rng])    # Char. cooling time [s]
    zT = np.array([np.minimum(0.5*delspr,0.5*v_g[i]*tauT[i,:,:]) for i in r0_rng])    # H_Tspr height [m]
    t_zT       = np.full_like(dmdr0,np.nan)    # Temperature at zT [K]
    wetdep_zT  = np.full_like(dmdr0,np.nan)    # Wetbulb depression at zT [K]
    # Heat flux due to size change
    zR_ig = np.where(np.logical_or(nospray,zerospray),np.nan,delspr/2)    # Initial guess for H_Rspr height [m]
    t_zR_ig = (th_0 - H_S0pr/G_S*np.log((z0t+zR_ig)/z0t))*th2t    # Initial guess for temp at zR_0 [K]
    q_zR_ig =   q_0 - H_L0pr/G_L*np.log((z0q+zR_ig)/z0q)    # Initial guess for q at zR_0 [kg kg-1]
    s_zR       = np.full_like(dmdr0,np.nan)    # Saturation ratio at zR [-]
    tauR       = np.full_like(dmdr0,np.nan)    # Characteristic evaporation time [s]
    zR         = np.full_like(dmdr0,np.nan)    # H_Rspr height [m]
    rf         = np.full_like(dmdr0,np.nan)    # Final droplet radius [m]
    
    # 4. Spray and spray-mediated heat fluxes ==============================================
    ET       = np.full_like(dmdr0,np.nan)    # Drop-specific efficiency for HT due to temp change [-]
    ER       = np.full_like(dmdr0,np.nan)    # Drop-specific efficiency for HT due to size change[-]
    H_Tspr   = np.where(nospray,np.nan,0)    # Spray heat flux due to temp change [W m-2]
    H_Sspr   = np.where(nospray,np.nan,0)    # Spray sensible heat flux [W m-2]
    H_Rspr   = np.where(nospray,np.nan,0)    # Spray heat flux due to size change [W m-2]
    H_Lspr   = np.where(nospray,np.nan,0)    # Spray latent heat flux [W m-2]
    H_Ssprpr = np.where(nospray,np.nan,0)    # Spray sensible heat flux, no feedback [W m-2]
    H_Rsprpr = np.where(nospray,np.nan,0)    # Spray heat flux due to size change, no feedback [W m-2]
    H_Lsprpr = np.where(nospray,np.nan,0)    # Spray latent heat flux, no feedback [W m-2]
    H_Rspr_IG = np.full_like(eps,np.nan)    # If fdbkfsolve == 'iterIG', the IG for H_Rspr [W m-2]
    ET_bar   = np.full_like(eps,np.nan)    # Drop-integrated efficiency for HT due to temp change [-]
    ER_bar   = np.full_like(eps,np.nan)    # Drop-integrated efficiency for HT due to size change [-]
    H_S0     = np.copy(H_S0pr)    # Interfacial sensible heat flux [W m-2]
    H_L0     = np.copy(H_L0pr)    # Interfacial latent heat flux [W m-2]
    H_S1     = np.where(zerospray,H_S0,np.nan)    # Total spray-mediated sensible heat flux [W m-2]
    H_L1     = np.where(zerospray,H_L0,np.nan)    # Total spray-mediated latent heat flux [W m-2]
    SP = ~np.logical_or(nospray,zerospray)    # Points where we will make calculations for spray
    SPindx = np.where(SP)    # Indices of SP
    
    print('Total spray points: %d' % np.sum(SP))
    for j in range(np.sum(SP)):

        print(j)
        spraymedHF_params = (r0_rng,r0,delta_r0,v_g,\
                np.array([zT[i,:,:][SP][j] for i in r0_rng]),\
                np.array([tauf[i,:,:][SP][j] for i in r0_rng]),\
                np.array([tauT[i,:,:][SP][j] for i in r0_rng]),\
                np.array([t_zR_ig[SP][j] for i in r0_rng]),\
                np.array([q_zR_ig[SP][j] for i in r0_rng]),\
                np.array([Fp[i,:,:][SP][j] for i in r0_rng]),\
                np.array([dmdr0[i,:,:][SP][j] for i in r0_rng]),\
                p_0[SP][j],gammaWB[SP][j],y0[SP][j],t_0[SP][j],rho_a[SP][j],\
                Dv_a[SP][j],xs[SP][j],delspr[SP][j],z0t[SP][j],z0q[SP][j],L[SP][j],\
                th_0[SP][j],q_0[SP][j],G_S[SP][j],G_L[SP][j],H_S0pr[SP][j],H_L0pr[SP][j],th2t[SP][j],\
                a_T[SP][j],a_R[SP][j],gamma_S[SP][j],gamma_L[SP][j],M_spr[SP][j],\
                Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,cp_sw,zRvaries)
        # Heat fluxes without feedback
        spraymedHF_rtrn = update_spraymedHF(0.0,0.0,0.0,spraymedHF_params)
        H_Ssprpr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[1]
        H_Rsprpr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[2]
        H_Lsprpr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[3]
        # Heat fluxes with feedback
        if feedback:
            # Calculate initial guess using simple model
            etaT = 17.502*240.97/(t_delsprD2pr[SP][j]-273.15+240.97)**2    # [K-1]
            Cs_pr = (1+y0[SP][j]-s_delsprD2pr[SP][j])**2/(1-s_delsprD2pr[SP][j])    # [-]
            if delspr[SP][j] < 4:
                C_HIG = 1.0    # Tuneable constant for equivalent height
            elif delspr[SP][j] > 10:
                C_HIG = 0.7
            else:
                C_HIG = -0.05*delspr[SP][j] + 1.2
            H_IG = min(C_HIG*delspr[SP][j],z_1[SP][j])    # Equivalent height for heating in simple model [m]
            Psi = etaT*np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*cp_a*kappa*ustar[SP][j])*spraymedHF_rtrn[0]    # [-]
            Chi = etaT*np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*cp_a*kappa*ustar[SP][j]) \
                     + np.log(z_1[SP][j]/H_IG)/(rho_a[SP][j]*Lv*kappa*ustar[SP][j]*q_delsprD2pr[SP][j])    # [m2 W-1]
            Lambda = (Psi - 1 + (1+y0[SP][j])/s_delsprD2pr[SP][j])/Chi    # [W m-2]
            A = spraymedHF_rtrn[2]/Cs_pr + 1/s_delsprD2pr[SP][j]/Chi    # [W m-2]
            B = -Lambda - y0[SP][j]/s_delsprD2pr[SP][j]/Chi    # [W m-2]
            C = y0[SP][j]*Lambda    # [W m-2]
            s_hatPOS = (-B + np.sqrt(B**2 - 4*A*C))/2/A    # Larger root [-], seems physical
            #s_hatNEG = (-B - np.sqrt(B**2 - 4*A*C))/2/A    # Smaller root [-], seems unphysical
            H_Rspr_IGj = s_hatPOS**2/(s_hatPOS-y0[SP][j])*spraymedHF_rtrn[2]/Cs_pr    # IG for H_Rspr [W m-2]
            if np.isnan(H_Rspr_IGj):
                H_Rspr_IGj = 0
            # Solve for feedback using chosen method
            if fdbkfsolve == 'fsolve':    # Solve feedback using fsolve
                # Solve for spray heat fluxes with feedback
                #IG = (spraymedHF_rtrn[1],spraymedHF_rtrn[2],spraymedHF_rtrn[3])    # Use fluxes without feedback as IG
                IG = (spraymedHF_rtrn[1],H_Rspr_IGj,H_Rspr_IGj + (spraymedHF_rtrn[0] - spraymedHF_rtrn[1]))    # Use simple model for IG
                H_spr_j,infodict,ier,mesg = fsolve(feedback_fsolve_residual,\
                        IG,args = spraymedHF_params,full_output = True)
                # Get other associated parameters.  If feedback did not converge, force NaNs in results.
                if ier != 1:
                    print('*** Feedback did not converge: j = %d ***' % j)
                    print('Fsolve failure flag: ier = %d' % ier)
                    print('Failure message: ' + mesg)
                    print('10m Neutral Windspeed: %f m/s' % U_10N[SP][j])
                    print('s_10N - y0: %f' % (s_10Npr[SP][j] - y0[SP][j]))
                    r0nans = np.full_like(r0,np.nan)    # Dummy SSGF-sized array of nans
                    spraymedHF_rtrn = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,\
                                       r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,r0nans,\
                                       np.nan,np.nan)    # Dummy rtrn of nans
                else:
                    spraymedHF_rtrn = update_spraymedHF(H_spr_j[0],H_spr_j[1],H_spr_j[2],spraymedHF_params)
            elif fdbkfsolve in ['iterNoIG','iterIG']:    # Perform a set number of iterations
                if fdbkfsolve == 'iterNoIG':    # Use fluxes without feedback as IG
                    N_iter_fdbk = 5    # Number of iterations to perform
                    H_Sspr_n = spraymedHF_rtrn[1]
                    H_Rspr_n = spraymedHF_rtrn[2]
                    H_Lspr_n = spraymedHF_rtrn[3]
                elif fdbkfsolve == 'iterIG':    # Use simple model to determine feedback IG
                    N_iter_fdbk = 1
                    H_Sspr_n = spraymedHF_rtrn[1]
                    H_Rspr_n = H_Rspr_IGj
                    H_Lspr_n = H_Rspr_IGj + (spraymedHF_rtrn[0] - spraymedHF_rtrn[1])
                # Perform selected number of iterations
                for n in range(N_iter_fdbk):
                    spraymedHF_rtrn = update_spraymedHF(H_Sspr_n,H_Rspr_n,H_Lspr_n,spraymedHF_params)
                    H_Sspr_n = spraymedHF_rtrn[1]
                    H_Rspr_n = spraymedHF_rtrn[2]
                    H_Lspr_n = spraymedHF_rtrn[3]
        H_Tspr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[0]
        H_Sspr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[1]
        H_Rspr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[2]
        H_Lspr[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[3]
        H_S0[SPindx[0][j],SPindx[1][j]]   = spraymedHF_rtrn[4]
        H_L0[SPindx[0][j],SPindx[1][j]]   = spraymedHF_rtrn[5]
        H_S1[SPindx[0][j],SPindx[1][j]]   = spraymedHF_rtrn[6]
        H_L1[SPindx[0][j],SPindx[1][j]]   = spraymedHF_rtrn[7]
        ET_bar[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[16]
        ER_bar[SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[17]
        if feedback and fdbkfsolve == 'iterIG':
            H_Rspr_IG[SPindx[0][j],SPindx[1][j]] = H_Rspr_IGj
        for i in r0_rng:
            t_zT[i,SPindx[0][j],SPindx[1][j]]      = spraymedHF_rtrn[8][i]
            wetdep_zT[i,SPindx[0][j],SPindx[1][j]] = spraymedHF_rtrn[9][i]
            s_zR[i,SPindx[0][j],SPindx[1][j]]      = spraymedHF_rtrn[10][i]
            tauR[i,SPindx[0][j],SPindx[1][j]]      = spraymedHF_rtrn[11][i]
            zR[i,SPindx[0][j],SPindx[1][j]]        = spraymedHF_rtrn[12][i]
            rf[i,SPindx[0][j],SPindx[1][j]]        = spraymedHF_rtrn[13][i]
            ET[i,SPindx[0][j],SPindx[1][j]]        = spraymedHF_rtrn[14][i]
            ER[i,SPindx[0][j],SPindx[1][j]]        = spraymedHF_rtrn[15][i]

    # 5. Output diagnostic quantities =======================================================
    if feedback == False:
        t_delsprD2 = np.copy(t_delsprD2pr)    # Temp at delspr/2 [K]
        q_delsprD2 = np.copy(q_delsprD2pr)    # q at delspr/2 [kg kg-1]
    elif feedback == True:
        t_delsprD2 = (th_0 - 1/G_S*(H_S0*(np.log(delspr/2/z0t) - psiH_delsprD2) \
                + 0.5*(1 - phisprH_delsprD2)*(H_Sspr - H_Rspr)))*th2t
        q_delsprD2 =   q_0 - 1/G_L*(H_L0*(np.log(delspr/2/z0q) - psiH_delsprD2) \
                + 0.5*(1 - phisprH_delsprD2)*H_Lspr)
    s_delsprD2 = satratio(t_delsprD2,p_0,q_delsprD2,0.99999)    # s at delspr/2 [-]
    delt_delsprD2 = t_delsprD2 - t_delsprD2pr    # Temp change at mid-layer due to feedback [K], (+) = warming
    delq_delsprD2 = q_delsprD2 - q_delsprD2pr    # q change at mid-layer due to feedback [kg kg-1], (+) = moistening
    dels_delsprD2 = s_delsprD2 - s_delsprD2pr    # s change at mid-layer due to feedback [-], (+) = incr s
    delt_delsprD2[zerospray] = 0
    delq_delsprD2[zerospray] = 0
    dels_delsprD2[zerospray] = 0
    if fdbkcrzyOPT == 0:    # Eliminate no points
        fdbkcrzy = np.full_like(delt_delsprD2,False,dtype=bool)
    elif fdbkcrzyOPT == 1:
        fdbkcrzy = np.logical_or(np.abs(delq_delsprD2) > 0.0005,np.abs(delt_delsprD2) > 2.)
    H_Tspr[fdbkcrzy] = np.nan
    H_Sspr[fdbkcrzy] = np.nan
    H_Rspr[fdbkcrzy] = np.nan
    H_Lspr[fdbkcrzy] = np.nan
    H_S0[fdbkcrzy] = np.nan
    H_L0[fdbkcrzy] = np.nan
    H_S1[fdbkcrzy] = np.nan
    H_L1[fdbkcrzy] = np.nan
    ET_bar[fdbkcrzy] = np.nan
    ER_bar[fdbkcrzy] = np.nan
    delt_delsprD2[fdbkcrzy] = np.nan
    delq_delsprD2[fdbkcrzy] = np.nan
    dels_delsprD2[fdbkcrzy] = np.nan
    alpha_S = H_Sspr/H_Ssprpr    # Spray feedback coefficient [-]
    beta_S = H_Rspr/H_Rsprpr    # Spray feedback coefficient [-]
    beta_L = H_Lspr/H_Lsprpr    # Spray feedback coefficient [-]
    beta_S[np.abs(beta_S) == np.inf] = np.nan
    beta_L[np.abs(beta_L) == np.inf] = np.nan
    Ch10N   =   H_S1/(rho_a*cp_a*U_10N*(t_0 - t_10Npr))    # 10m neutral SH transfer coefficient [-]
    Ch10Npr = H_S0pr/(rho_a*cp_a*U_10N*(t_0 - t_10Npr))    # 10m neutral SH transfer coefficient without spray [-]
    Cq10N   =   H_L1/(rho_a*Lv*U_10N*(q_0 - q_10Npr))    # 10m neutral LH transfer coefficient [-]
    Cq10Npr = H_L0pr/(rho_a*Lv*U_10N*(q_0 - q_10Npr))    # 10m neutral LH transfer coefficient without spray [-]
    Ck10N   =   (H_S1 +   H_L1)/(rho_a*U_10N*(cp_a*(t_0 - t_10Npr) + Lv*(q_0 - q_10Npr)))    # 10N enth coeff [-]
    Ck10Npr = (H_S0pr + H_L0pr)/(rho_a*U_10N*(cp_a*(t_0 - t_10Npr) + Lv*(q_0 - q_10Npr)))    # 10N enth coeff w/o spray [-]
    if showfdbkcrzy:
        delt_delsprD2[fdbkcrzy] = 2.
        delq_delsprD2[fdbkcrzy] = 0.0005
        dels_delsprD2[fdbkcrzy] = 0.05

    # 6. Vertical profiles (if requested) ===================================================
    if getprofiles == False:
        profiles = None
    elif getprofiles == True:
        # Initialize lists
        z_t = []    # z-values counting from z0t [m]
        z_q = []    # z-values counting from z0q [m]
        th_prof = []    # Vertical potential temperature profiles [K]
        q_prof = []    # Vertical specific humidity profiles [kg kg-1]
        # 1. Roughness heights
        z_t.append(z0t)
        z_q.append(z0q)
        th_prof.append(th_0)
        q_prof.append(q_0)
        # 2. Spray layer
        nzspr = 50    # Number of points in spray layer
        zspr_inc = np.linspace(0,1,nzspr)[1:-1]    # Values at z0h and delspr added separately
        for k in zspr_inc:
            zK = 10**(-7+k*(np.log10(delspr)+7))    # These zK are relative to z0t or z0q
            z_t.append(z0t+zK)
            z_q.append(z0q+zK)
            if feedback == False:
                th_prof.append(th_0 - 1/G_S*(H_S0*(np.log((z0t+zK)/z0t) - stabIntH((z0t+zK)/L))))
                q_prof.append(  q_0 - 1/G_L*(H_L0*(np.log((z0q+zK)/z0q) - stabIntH((z0q+zK)/L))))
            elif feedback == True:
                th_prof.append(th_0 - 1/G_S*(H_S0*(np.log((z0t+zK)/z0t) - stabIntH((z0t+zK)/L)) + zK/delspr*(1 - stabIntSprayH((z0t+zK)/L))*(H_Sspr - H_Rspr)))
                q_prof.append(  q_0 - 1/G_L*(H_L0*(np.log((z0q+zK)/z0q) - stabIntH((z0q+zK)/L)) + zK/delspr*(1 - stabIntSprayH((z0q+zK)/L))*H_Lspr))
        # 3. Above spray layer
        nzabv = 20    # Number of points above spray layer
        zabv_inc = np.linspace(0,1,nzabv)[:-1]    # Values at z_1 are added separately
        for k in zabv_inc:
            zK = delspr + k*(z_1 - delspr)    # These zK are absolute heights
            z_t.append(zK)
            z_q.append(zK)
            if feedback == False:
                th_prof.append(th_1 + H_S0pr/G_S*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
                q_prof.append(  q_1 + H_L0pr/G_L*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
            elif feedback == True:
                th_prof.append(th_1 +   H_S1/G_S*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
                q_prof.append(  q_1 +   H_L1/G_L*(np.log(z_1/zK) - (psiH_1 - stabIntH(zK/L))))
        # 4. Lowest model level
        z_t.append(z_1)
        z_q.append(z_1)
        th_prof.append(th_1)
        q_prof.append(q_1)
        # Gather data for outputting
        z_t     = np.array(z_t)
        z_q     = np.array(z_q)
        t_prof  = np.array([th*th2t for th in th_prof])
        th_prof = np.array(th_prof)
        q_prof  = np.array(q_prof)
        sMy0_prof = np.array([satratio(t_prof[k,:,:],p_0,q_prof[k,:,:],0.99999) - y0 \
                for k in range(np.shape(z_t)[0])])    # s(z) minus y0 [-]
        for k in range(np.shape(z_t)[0]):
            z_t[k,:,:][~SP] = np.nan
            z_q[k,:,:][~SP] = np.nan
            t_prof[k,:,:][~SP] = np.nan
            th_prof[k,:,:][~SP] = np.nan    # Since we don't calculate p(z), t is conserved in adiabatic lifting, so no need to output th_prof
            q_prof[k,:,:][~SP] = np.nan
            sMy0_prof[k,:,:][~SP] = np.nan
        profiles = [z_t,z_q,t_prof - 273.15,q_prof*1000,sMy0_prof]

    # Return analysis results ===================================================================
    szRMy0 = np.array([s_zR[i,:,:] - y0 for i in r0_rng])    # s_zR minus y0 [-]
    return [H_S0,H_L0,H_S1,H_L1,H_Tspr,H_Sspr,H_Rspr,H_Lspr,alpha_S,beta_S,beta_L,gamma_S,t_zT,\
            delt_delsprD2,delq_delsprD2,r0,M_spr,dmdr0/1e6,a_T,a_R,ET_bar,ER_bar,ET,ER,tauR,profiles,zR,\
            rf*1e6,szRMy0,zT,wetdep_zT,U_10N,t_10Npr,wetdep_10Npr,q_10Npr,s_10Npr - y0,H_S0pr,H_L0pr,\
            q_0 - q_10Npr,Ch10N,Cq10N,Ck10N,Ch10Npr,Cq10Npr,Ck10Npr,delta_r0,dels_delsprD2,U_10,gamma_L,\
            H_Rspr_IG]

def sprayHF(r0,delta_r0,dmdr0,t_zT,tdropf,tWB_zT,rf,a_T,a_R,t_0,M_spr,cp_sw,Lv):
    """
    Calculate spray heat fluxes for a single gridpoint based on provided thermodynamic and 
    SSGF properties.
                                 |-------|
    Thermodynamic parameters --> |sprayHF| --> Spray Heat Fluxes
                                 |-------|
    """
    ET = 1/a_T*cp_sw*(t_0 - tdropf)    # Efficiency for HT due to temp change [-]
    ES = 1/a_T*cp_sw*np.sign(t_0 - tWB_zT)*np.minimum(np.abs(t_0 - tdropf),np.abs(t_0 - t_zT))    # Eff for SHF [-]
    ER = 1/a_R*Lv*(1 - (rf/r0)**3)    # Efficiency for HT due to size change [-]
    H_Tspr = a_T*np.dot(ET*dmdr0,delta_r0)    # Spray HF due to temp change [W m-2]
    H_Sspr = a_T*np.dot(ES*dmdr0,delta_r0)    # Spray SHF [W m-2]
    H_Rspr = a_R*np.dot(ER*dmdr0,delta_r0)    # Spray HF due to size change [W m-2]
    H_Lspr = H_Rspr + H_Tspr - H_Sspr    # Spray LHF [W m-2]
    ET_bar = H_Tspr/a_T/M_spr    # Mean efficiency for HT due to temp change [-]
    ER_bar = H_Rspr/a_R/M_spr    # Mean efficiency for HT due to size change [-]
    return (ET,ER,H_Tspr,H_Sspr,H_Rspr,H_Lspr,ET_bar,ER_bar)

def update_spraymedHF(H_Sspr_0,H_Rspr_0,H_Lspr_0,spraymedHF_params):
    """
    Update spray and spray-mediated heat fluxes for a single gridpoint using a previous
    set of values for spray heat fluxes.
                          |-----------------|
    Spray Heat Fluxes --> |update_spraymedHF| --> All Heat Fluxes
                          |-----------------|
    """
    r0_rng,r0,delta_r0,v_g,zT,tauf,tauT,t_zR_ig,q_zR_ig,Fp,dmdr0,\
            p_0,gammaWB,y0,t_0,rho_a,Dv_a,xs,delspr,z0t,z0q,L,\
            th_0,q_0,G_S,G_L,H_S0pr,H_L0pr,th2t,a_T,a_R,gamma_S,gamma_L,M_spr,\
            Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,cp_sw,zRvaries = spraymedHF_params
    # Surface fluxes using previous values of spray heat fluxes
    H_S1_0 = H_S0pr + gamma_S*(H_Sspr_0 - H_Rspr_0)
    H_L1_0 = H_L0pr + gamma_L*H_Lspr_0
    H_S0_0 = H_S1_0 - (H_Sspr_0 - H_Rspr_0)
    H_L0_0 = H_L1_0 - H_Lspr_0
    # Thermodynamic parameters for H_Tspr
    t_zT,wetdep_zT,tWB_zT,tdropf = thermo_HTspr(zT,tauf,tauT,p_0,gammaWB,y0,t_0,\
            delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0_0,H_L0_0,H_Sspr_0,H_Rspr_0,H_Lspr_0,th2t,Lv,cp_a)
    # Thermodynamic parameters for H_Rspr
    s_zR,tauR,zR,rf = thermo_HRspr(r0_rng,r0,v_g,t_zR_ig,q_zR_ig,Fp,tauf,p_0,gammaWB,y0,t_0,rho_a,Dv_a,xs,\
            delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0_0,H_L0_0,\
            H_Sspr_0,H_Rspr_0,H_Lspr_0,th2t,Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,zRvaries)
    # Updated spray heat fluxes
    ET,ER,H_Tspr,H_Sspr,H_Rspr,H_Lspr,ET_bar,ER_bar = sprayHF(r0,delta_r0,dmdr0,t_zT,tdropf,tWB_zT,rf,\
            a_T,a_R,t_0,M_spr,cp_sw,Lv)
    # Updated surface and spray-mediated heat fluxes
    H_S1 = H_S0pr + gamma_S*(H_Sspr - H_Rspr)
    H_L1 = H_L0pr + gamma_L*H_Lspr
    H_S0 = H_S1 - (H_Sspr - H_Rspr)
    H_L0 = H_L1 - H_Lspr
    return (H_Tspr,H_Sspr,H_Rspr,H_Lspr,H_S0,H_L0,H_S1,H_L1,t_zT,wetdep_zT,s_zR,tauR,zR,rf,ET,ER,ET_bar,ER_bar)

def feedback_fsolve_residual(H_spr,*spraymedHF_params):
    """
    Residuals for spray heat fluxes, used by fsolve in spraymedHF.
    Performs calculations for a single gridpoint.
    """
    H_Sspr_0,H_Rspr_0,H_Lspr_0 = H_spr
    spraymedHF_rtrn = update_spraymedHF(H_Sspr_0,H_Rspr_0,H_Lspr_0,spraymedHF_params)
    Res_S = spraymedHF_rtrn[1] - H_Sspr_0    # Residual for H_Sspr [W m-2]
    Res_R = spraymedHF_rtrn[2] - H_Rspr_0    # Residual for H_Rspr [W m-2]
    Res_L = spraymedHF_rtrn[3] - H_Lspr_0    # Residual for H_Lspr [W m-2]
    return (Res_S,Res_R,Res_L)

