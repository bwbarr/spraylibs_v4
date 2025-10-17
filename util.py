import numpy as np
from scipy.optimize import fsolve

def fall_velocity_PK97(r):
    """
    Fall velocity of spherical droplets, based on Pruppacher and Klett (1997) section 10.3.6.
    Parameters:
        r - droplet radius [m]
    Return:
        v_fall - settling velocity [m s-1]
    """
    nu_a = 1.5e-5    # Kinematic viscosity of air [m2 s-1]
    rho_a = 1.25    # Density of air [kg m-3]
    rho_w = 1030    # Density of seawater [kg m-3]
    g = 9.81    # Acceleration due to gravity [m s-2]
    sigma_aw = 7.4e-2    # Surface tension of air-water interface [N m-1]
    v_stokes = 2*r**2*g*(rho_w - rho_a)/9/(rho_a*nu_a)    # Stokes velocity [m s-1]
    
    # r < 10 micrometers
    lamda_a0 = 6.6e-8    # Mean free path at 1013.25 mb, 293.15 K [m]
    f_slip = 1 + 1.26*lamda_a0/r    # Slip flow Cunningham correction factor [-]
    v_small = f_slip*v_stokes    # Settling velocity for r < 10 micrometers [m s-1]
    
    # 10 <= r <= 535 micrometers
    CdNRe2 = 32*r**3*(rho_w - rho_a)/rho_a/nu_a**2*g/3    # Product of drag coefficient and square of Reynolds number [-]
    X = np.log(CdNRe2)    # X in polynomial curve fit [-]
    B = np.array([-0.318657e1, 0.992696, -0.153193e-2, -0.987059e-3, -0.578878e-3, 0.855176e-4, -0.327815e-5])    # Polynomial coefficients [-]
    Y = B[0] + B[1]*X + B[2]*X**2 + B[3]*X**3 + B[4]*X**4 + B[5]*X**5 + B[6]*X**6    # Y in polynomial curve fit [-]
    NRe = np.exp(Y)    # Reynolds number [-]
    v_med = nu_a*NRe/2/r    # Settling velocity for 10 <= r <= 535 micrometers [m s-1]
    
    # r > 535 micrometers
    NBo = g*(rho_w - rho_a)*r**2/sigma_aw    # Bond number [-]
    NP = sigma_aw**3/rho_a**2/nu_a**4/g/(rho_w - rho_a)    # Physical property number [-]
    NBoNP16 = NBo*NP**(1/6)    # Product of Bond number and physical property number to the 1/6 power [-]
    X = np.log(16/3*NBoNP16)    # X in polynomial curve fit [-]
    B = np.array([-0.500015e1, 0.523778e1, -0.204914e1, 0.475294, -0.542819e-1, 0.238449e-2])    # Polynomial coefficients [-]
    Y = B[0] + B[1]*X + B[2]*X**2 + B[3]*X**3 + B[4]*X**4 + B[5]*X**5    # Y in polynomial curve fit [-]
    NRe = NP**(1/6)*np.exp(Y)    # Reynolds number [-]
    v_large = nu_a*NRe/2/r    # Settling velocity for r > 535 micrometers [m s-1]
    
    # Connect regimes
    v_fall = v_med
    v_fall[r < 10e-6] = v_small[r < 10e-6]
    v_fall[r > 535e-6] = v_large[r > 535e-6]
    return v_fall

def qsat0(T_K,P):
    """
    Saturation specific humidity over a plane surface of pure water, with e_sat0 per Buck (1981) Eq 8.
    Parameters:
        T_K - temperature [K]
        P - pressure [Pa]
    Return:
        q_sat0 - saturation specific humidity [kg kg-1]
    """
    e_sat0 = 6.1121*np.exp(17.502*(T_K - 273.15)/(T_K - 273.15 + 240.97))*(1.0007 + 3.46e-8*P)*1e2    # Sat vap press [Pa]
    q_sat0 = e_sat0*0.622/(P - 0.378*e_sat0)    # Saturation specific humidity [kg kg-1]
    return q_sat0

def satratio(T_K,P,q,max_satratio):
    """
    Saturation ratio, with e_sat0 per Buck (1981) Eq 8.  We assume that s = q/qsat, rather than w/wsat, so that air with q = qsat
    will be saturated.
    Parameters:
        T_K - temperature [K]
        P - pressure [Pa]
        q - specific humidity [kg kg-1]
        max_satratio - maximum allowable saturation ratio [-]
    Return:
        s - saturation ratio [-]
    """
    q_sat0 = qsat0(T_K,P)    # Saturation specific humidity [kg kg-1]
    s = np.minimum(q/q_sat0,max_satratio)    # Saturation ratio [-]
    return s   

def stabIntH(zeta):
    """
    Integrated stability function for heat, evaluated at zeta.  Per Garratt (1992).
    Parameters:
        zeta - stability parameter [-]
    Return:
        psiH - integrated stability function at zeta [-]
    """
    psiH = np.full_like(zeta,np.nan)
    psiH[zeta == 0] = 0
    psiH[zeta > 0] = -5*zeta[zeta > 0]
    Y = (1 - 16*zeta[zeta < 0])**0.5
    psiH[zeta < 0] = 2*np.log((1 + Y)/2)
    return psiH

def stabIntM(zeta):
    """
    Integrated stability function for momentum, evaluated at zeta.  Per Garratt (1992).
    Parameters:
        zeta - stability parameter [-]
    Return:
        psiM - integrated stability function at zeta [-]
    """
    psiM = np.full_like(zeta,np.nan)
    psiM[zeta == 0] = 0
    psiM[zeta > 0] = -5*zeta[zeta > 0]
    X = (1 - 16*zeta[zeta < 0])**0.25
    psiM[zeta < 0] = 2*np.log((1 + X)/2) + np.log((1 + X**2)/2) - 2*np.arctan(X) + np.pi/2
    return psiM

def stabIntSprayH(zeta):
    """
    Integrated stability function for heat within spray layer, evaluated at zeta.
    Parameters:
        zeta - stability parameter [-]
    Return:
        phisprH - integrated stability function within spray layer at zeta [-]
    """
    phisprH = np.full_like(zeta,np.nan)
    phisprH[zeta == 0] = 0
    phisprH[zeta > 0] = -2.5*zeta[zeta > 0]
    Y = (1 - 16*zeta[zeta < 0])**0.5
    phisprH[zeta < 0] = -(Y - 1)**2/16/zeta[zeta < 0]
    return phisprH

def thermo_HRspr(r0_rng,r0,v_g,t_zR_ig,q_zR_ig,Fp,tauf,p_0,gammaWB,y0,t_0,rho_a,Dv_a,xs,delspr,z0t,z0q,L,\
        th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,th2t,Lv,cp_a,rho_sw,nu,Phi_s,Mw,Ms,zRvaries):
    """
    Droplet thermodynamic calculations for spray heat flux due to size change.
    Performs calculations for the entire droplet size vector.
                        |------------|
    All Heat Fluxes --> |thermo_HRspr| --> Thermodynamic parameters
                        |------------|
    """
    t_zR = np.full_like(t_zR_ig,np.nan)
    q_zR = np.full_like(t_zR_ig,np.nan)
    fsolveFailed = np.full_like(t_zR_ig,False,dtype=bool)    # True if fsolve failed to find a solution
    t_zR_const = (th_0 - 1/G_S*(H_S0*(np.log((z0t+delspr/2)/z0t) - stabIntH((z0t+delspr/2)/L)) \
            + 0.5*(1 - stabIntSprayH((z0t+delspr/2)/L))*(H_Sspr - H_Rspr)))*th2t    # t_zR if zR is constant [K]
    q_zR_const =   q_0 - 1/G_L*(H_L0*(np.log((z0q+delspr/2)/z0q) - stabIntH((z0q+delspr/2)/L)) \
            + 0.5*(1 - stabIntSprayH((z0q+delspr/2)/L))*H_Lspr)    # q_zR if zR is constant [kg kg-1]
    for i in r0_rng:
        if r0[i] < 100*1e-6 and zRvaries == True:    # Calculate zR iteratively
            tq_zR_fsolve_params = (r0[i],v_g[i],Fp[i],p_0,gammaWB,y0,rho_a,Dv_a,delspr,z0t,z0q,L,th_0,\
                    q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,th2t,Lv,cp_a,rho_sw)
            tq_zR_i,infodict,ier,mesg = fsolve(tq_zR_fsolve_residual,(t_zR_ig[i],q_zR_ig[i]),\
                    args = tq_zR_fsolve_params,full_output = True)
            t_zR[i] = tq_zR_i[0]
            q_zR[i] = tq_zR_i[1]
            fsolveFailed[i] = False if ier == 1 else True
        else:
            t_zR[i] = t_zR_const
            q_zR[i] = q_zR_const
    qsat0_zR = qsat0(t_zR,p_0)    # Saturation specific humidity at zR [kg kg-1]
    betaWB_zR = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zR)    # Wetbulb coefficient at zR [-]
    s_zR = satratio(t_zR,p_0,q_zR,0.99999)    # Saturation ratio at zR [-]
    tauR = rho_sw*r0**2/(rho_a*Dv_a*Fp*qsat0_zR*betaWB_zR*np.abs(1 + y0 - s_zR))    # Char timescale for evap [s]
    req = r0*(xs*(1 + nu*Phi_s*Mw/Ms/(1 - s_zR)))**(1/3)    # Equilibrium radius at zR [m]
    delR = v_g*tauR    # Layer thickness governing H_Rspr [m]
    if zRvaries:
        zR = np.minimum(0.5*delspr,0.5*delR)    # H_Rspr height [m]
    else:
        zR = delspr/2
    rf = req + (r0 - req)*np.exp(-tauf/tauR)    # Final droplet radius [m]
    # Replace points where s_zR ~ 1 + y0, or fsolve fails
    szR_EQ_1Py0 = np.logical_and(abs(1 + y0 - s_zR) < 1e-3,~fsolveFailed)    # Points where s_zR ~ 1 + y0
    #szR_EQ_1Py0  = np.full_like(t_zR_ig,False,dtype=np.bool)    # Uncomment to keep these points; for debugging
    #fsolveFailed = np.full_like(t_zR_ig,False,dtype=np.bool)    # Uncomment to keep these points; for debugging
    s_zR = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0, 1 + y0,s_zR))
    tauR = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0, np.nan,tauR))
    zR   = np.where(fsolveFailed,np.nan,\
           np.where(szR_EQ_1Py0,\
               np.where(np.logical_and(abs(t_0 - t_zR) < 5e-1,abs(q_0 - q_zR) < 5e-4),0,np.nan),zR))
    rf   = np.where(fsolveFailed,r0,\
           np.where(szR_EQ_1Py0, r0,rf))
    return (s_zR,tauR,zR,rf)

def thermo_HTspr(zT,tauf,tauT,p_0,gammaWB,y0,t_0,delspr,z0t,z0q,L,\
        th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,H_Rspr,H_Lspr,th2t,Lv,cp_a):
    """
    Droplet thermodynamic calculations for spray heat flux due to temp change.
    Performs calculations for the entire droplet size vector.
                        |------------|
    All Heat Fluxes --> |thermo_HTspr| --> Thermodynamic parameters
                        |------------|
    """
    t_zT = (th_0 - 1/G_S*(H_S0*(np.log((z0t+zT)/z0t) - stabIntH((z0t+zT)/L)) \
            + zT/delspr*(1 - stabIntSprayH((z0t+zT)/L))*(H_Sspr - H_Rspr)))*th2t    # Temp at zT [K]
    q_zT =   q_0 - 1/G_L*(H_L0*(np.log((z0q+zT)/z0q) - stabIntH((z0q+zT)/L)) \
            + zT/delspr*(1 - stabIntSprayH((z0q+zT)/L))*H_Lspr)    # Spec hum at zT [kg kg-1]
    qsat0_zT = qsat0(t_zT,p_0)    # Saturation specific humidity at zT [kg kg-1]
    betaWB_zT = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zT)    # Wetbulb coefficient at zT [-]
    s_zT = satratio(t_zT,p_0,q_zT,0.99999)    # Saturation ratio at zT [-]
    wetdep_zT = (1 - s_zT/(1 + y0))*(1 - betaWB_zT)/gammaWB    # Wetbulb depression at zT [-]
    tWB_zT = t_zT - wetdep_zT    # Wetbulb temperature at zT [K]
    tdropf = tWB_zT + (t_0 - tWB_zT)*np.exp(-tauf/tauT)    # Final droplet temperature [K]
    return (t_zT,wetdep_zT,tWB_zT,tdropf)

def tq_zR_fsolve_residual(tq_zR,*params):
    """
    Residuals for profile equations for t_zR and q_zR, used by fsolve in thermo_HRspr.
    Performs calculations on one element of the droplet size vector.
    """
    t_zR,q_zR = tq_zR    # Temp and specific humidity at zR [K, kg kg-1]
    r0,v_g,Fp,p_0,gammaWB,y0,rho_a,Dv_a,delspr,z0t,z0q,L,th_0,q_0,G_S,G_L,H_S0,H_L0,H_Sspr,\
            H_Lspr,H_Rspr,th2t,Lv,cp_a,rho_sw = params
    qsat0_zR = qsat0(t_zR,p_0)    # Saturation specific humidity at zR [kg kg-1]
    betaWB_zR = 1/(1 + Lv*gammaWB*(1 + y0)/cp_a*qsat0_zR)    # Wetbulb coefficient at zR [-]
    s_zR = satratio(t_zR,p_0,q_zR,0.99999)    # Saturation ratio at zR [-]
    tauR = rho_sw*r0**2/(rho_a*Dv_a*Fp*qsat0_zR*betaWB_zR*np.abs(1 + y0 - s_zR))    # Char timescale for evap [s]
    delR = v_g*tauR    # Layer thickness governing H_Rspr [m]
    zR = np.minimum(0.5*delspr,0.5*delR)    # H_Rspr height [m]
    Res_t_zR = (th_0 - 1/G_S*(H_S0*(np.log((z0t+zR)/z0t) - stabIntH((z0t+zR)/L)) \
            + zR/delspr*(1 - stabIntSprayH((z0t+zR)/L))*(H_Sspr - H_Rspr)))*th2t - t_zR    # Res for t [K], O(1)
    Res_q_zR =  (q_0 - 1/G_L*(H_L0*(np.log((z0q+zR)/z0q) - stabIntH((z0q+zR)/L)) \
            + zR/delspr*(1 - stabIntSprayH((z0q+zR)/L))*H_Lspr) - q_zR)*1000    # Res for q [g kg-1], O(1)
    return (Res_t_zR,Res_q_zR)

