import numpy as np
from scipy.special import erf
from .util import fall_velocity_PK97,stabIntM

# Default SSGF r0 and delta_r0 vectors
r0_default       = np.array([10,20,30,40,50,60,70,80,90,102.5,122.5,157.5,215,300,400,500,600,700,800,900,1037.5,1250,1500,1750,2000])*1e-6    # [m]
delta_r0_default = np.array([10,10,10,10,10,10,10,10,10,   15,   25,   45, 70,100,100,100,100,100,100,100,   175, 250, 250, 250, 250])*1e-6    # [m]

def sourceM_F09_dev1(model_coeffs,eps,swh,dcp,mss,wspd10,ustar,z0,L,r0=None,delta_r0=None,chi1=1.0,chi2=1.0):
    """
    SSGF based on Fairall et al. 2009.  Returns the r0 vector, delta_r0 vector, 
    total spray mass flux array (y,x), and the droplet SSGF (r0,y,x).
    Parameters:
        model_coeffs - SSGF model coefficients
        eps - wave energy dissipation flux [kg s-3]
        swh - significant wave height [m]
        dcp - dominant phase velocity [m s-1]
        mss - mean squared waveslope [-]
        wspd10 - 10-m windspeed [m s-1]
        ustar - friction velocity [m s-1]
        z0 - momentum roughness length [m]
        L - Obukhov stability length [m]
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
        chi1 - factor scaling small droplet end of SSGF
        chi2 - factor scaling large droplet end of SSGF
    Return:
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
        M_spr - total spray mass flux [kg m-2 s-1]
        dmdr0 - mass spectrum of ejected droplets [kg m-2 s-1 m-1]
    """

    # Constants and intermediate variables
    fs = model_coeffs[0]    # Factor scaling spectra magnitude (this is Chris' sourcestrength) [-]
    C1 = model_coeffs[1]    # Model coefficient scaling magnitude (absorbs fd, the fraction of dissipated energy that forms droplets) [-]
    C2 = model_coeffs[2]    # Model coefficient inside exponential [-]
    C3 = model_coeffs[3]    # Model coefficient on mss [-]
    C4 = model_coeffs[4]    # Model coefficient on sigma_h [-]
    C5 = model_coeffs[5]    # Model coefficient in erf [-]
    Ceps = 100    # Nondimensional mean volumetric dissipation [-], from Sutherland and Melville 2015
    Cbr = 0.8    # Scale factor on dcp to determine breaking wave crest speed [-], ~0.8 per Banner et al. 2014
    alpha_k = 1.5    # Kolmogorov constant [-]
    kappa = 0.41    # von Karman constant [-]
    rho_sw = 1030.    # Density of seawater [kg m-3]
    nu_w = 0.90e-6    # Kinematic viscosity of water [m2 s-1]
    sigma_surf = 7.4e-5    # Ratio of surface tension to water density [m3 s-2]
    a1 = 0.56    # Coefficient in SSGF scaling function [-]
    a2 = 0.58    # Coefficient in SSGF scaling function [-]
    Wa = whitecapActive_DLM17(dcp,ustar,swh)    # Actively breaking whitecap fraction [-]
    eps_KV_mean = Ceps*eps/swh/rho_sw    # Mean surface volumetric kinematic dissipation [m2 s-3], per Sutherland and Melville 2015
    eps_KV = eps_KV_mean/Wa    # Surface volumetric kinematic dissipation under whitecaps [m2 s-3]
    eta_k = (nu_w**3/eps_KV)**0.25    # Kolmogorov microscale [m]
    h_gust = 200*z0    # Gust height [m]
    wspdh = ustar/kappa*(np.log(h_gust/z0) - stabIntM(h_gust/L))    # Windspeed at gust height [m s-1]
    U_crest = Cbr*dcp    # Speed of crest of breaking wave [m s-1]
    sigma_h = C4*wspd10    # Standard deviation of windspeed at h_gust [m s-1]
    if r0 is None:
        r0 = r0_default
        delta_r0 = delta_r0_default
    v_fall = fall_velocity_PK97(r0)    # Fall velocity of droplets [m s-1]

    # SSGF scaling function
    gscale = a1*np.log10(r0*1e6)-a2    # g function, requires r0 in um
    hscaleP = np.exp(-1/gscale)    # h function, "plus g" version
    hscaleM = np.exp(-1/(1-gscale))    # h function, "minus g" version
    fscale = (chi1*hscaleM + chi2*hscaleP)/(hscaleM + hscaleP)    # SSGF scaling function
    fscale[np.log10(r0*1e6) <= a2/a1+0.1] = chi1    # Set value below valid interval
    fscale[np.log10(r0*1e6) >= (1+a2)/a1-0.1] = chi2    # Set value above valid interval

    # Calculate spray mass spectrum
    dmdr0_form = np.array([(fs*C1*rho_sw*eps_KV*r*Wa)/(3*sigma_surf)*np.exp(-3/2*alpha_k*C2*(np.pi*eta_k/r)**(4/3))\
            for r in r0])    # Spectrum of droplets formed from wave breaking [kg m-2 s-1 m-1]
    ejecprob = np.array([(1. + erf((wspdh - U_crest - v/C3/mss)/sigma_h - C5))/2 \
            for v in v_fall])    # Droplet ejection probability [-]
    dmdr0 = np.array([dmdr0_form[i,:,:]*ejecprob[i,:,:]*fscale[i] for i in range(np.size(r0))])    # Ejected droplet spectrum [kg m-2 s-1 m-1]

    # Calculate total spray mass flux
    dims = np.shape(eps)
    M_spr = np.zeros_like(eps)
    for i in range(dims[0]):
        for j in range(dims[1]):
            M_spr[i,j] = np.dot(dmdr0[:,i,j],delta_r0)    # [kg m-2 s-1]

    return r0,delta_r0,M_spr,dmdr0

def sourceM_F09_prev(fs,eps,swh,dcp,mss,wspd10,wspdh,r0=None,delta_r0=None):
    """
    SSGF based on Fairall et al. 2009.  Returns the r0 vector, delta_r0 vector, 
    total spray mass flux array (y,x), and the droplet SSGF (r0,y,x).
    Parameters:
        fs - factor to scale spectra magnitude (this is Chris' sourcestrength) [-]
        eps - wave energy dissipation flux [kg s-3]
        swh - significant wave height [m]
        dcp - dominant phase velocity [m s-1]
        mss - mean squared waveslope [-]
        wspd10 - 10-m windspeed [m s-1]
        wspdh - windspeed at gust height [m s-1]
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
    Return:
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
        M_spr - total spray mass flux [kg m-2 s-1]
        dmdr0 - mass spectrum of ejected droplets [kg m-2 s-1 m-1]
    """

    # Constants and intermediate variables
    C1 = 3.0    # Experimentally-tuned coefficient [-]
    C2 = 2.87    # Chris' coefficient [-]
    C3 = 0.08    # Chris' coefficient [-]
    C4 = 1.0    # Experimentally-tuned coefficient [-]
    C7 = 1.0    # Experimentally-tuned coefficient [-]
    g = 9.81    # Acceleration due to gravity [m s-2]
    kappa = 0.41    # von Karman constant [-]
    alpha_k = 0.54    # Kolmogorov constant [-]
    rho_w = 1030.    # Density of seawater [kg m-3]
    nu_w = 0.90e-6    # Kinematic viscosity of water [m2 s-1]
    sigma_surf = 7.4e-5    # Ratio of surface tension to water density [m3 s-2]
    Lb = 1.0    # Depth scale for breaking wave turbulence [m]
    WB_Melville = np.minimum(2.0e-7*wspd10**3.4,1.0)    # Active breaking whitecap fraction, per Melville [-]
    eta_k = (rho_w*Lb*WB_Melville/eps*nu_w**3)**0.25    # Kolmogorov microscale [m]
    sigma_h = C7*wspd10    # Standard deviation of windspeed at h_gust [m s-1]
    if r0 is None:
        r0 = r0_default
        delta_r0 = delta_r0_default
    v_fall = fall_velocity_PK97(r0)    # Fall velocity of droplets [m s-1]

    # Calculate spray mass spectrum
    dmdr0_form = np.array([(C1*fs*eps*r)/(sigma_surf*Lb)*np.exp(-9/4*C2*alpha_k*(C3*np.pi*eta_k/r)**(4/3))\
            for r in r0])    # Spectrum of droplets formed from wave breaking [kg m-2 s-1 m-1]
    ejecprob = np.array([(1. + erf((wspdh - dcp/2 + g*swh/2/dcp - v/C4/mss)/sigma_h))/2 \
            for v in v_fall])    # Droplet ejection probability [-]
    dmdr0 = dmdr0_form*ejecprob    # Ejected droplet spectrum [kg m-2 s-1 m-1]

    # Calculate total spray mass flux
    dims = np.shape(eps)
    M_spr = np.zeros_like(eps)
    for i in range(dims[0]):
        for j in range(dims[1]):
            M_spr[i,j] = np.dot(dmdr0[:,i,j],delta_r0)    # [kg m-2 s-1]

    return r0,delta_r0,M_spr,dmdr0

def sourceM_F94(fs,wspd10,Wform,r0=None,delta_r0=None):
    """
    SSGF per Fairall et al. 1994.  Implemented per Mueller and Veron (2014b).
    Returns the r0 vector, delta_r0 vector, total spray mass flux array (y,x), and the droplet SSGF (r0,y,x).
    Parameters:
        fs - factor to scale spectra magnitude (this is Chris' sourcestrength) [-]
        wspd10 - 10-m windspeed [m s-1]
        Wform - formulation for whitecap fraction: MOM80 for Monahan & O'Muir 1980; CF21 for form received from Chris on 210504
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
    Return:
        r0 - SSGF radius vector [m]
        delta_r0 - SSGF bin width vector [m]
        M_spr - total spray mass flux [kg m-2 s-1]
        dmdr0 - mass spectrum of ejected droplets [kg m-2 s-1 m-1]
    """
    rho_w = 1030.    # Density of seawater [kg m-3]
    if r0 is None:
        r0 = r0_default
        delta_r0 = delta_r0_default
    r0_micrometers = r0*1e6    # [\mu m]
    r80 = 0.518*r0_micrometers**0.976    # Equilibrium radius at 80% RH [\mu m], per Fitzgerald (1975)
    
    # Coefficients for A92 source function at 11 m/s, based on r80
    B0 = 4.405
    B1 = -2.646
    B2 = -3.156
    B3 = 8.902
    B4 = -4.482
    C1 = 1.02e4
    C2 = 6.95e6
    C3 = 1.75e17
    
    dFdr80 = np.zeros_like(r80)    # A92 source function at 11 m/s, based on r80 [m-2 s-1 (\mu m)-1]
    dFdr80 = np.where(np.logical_and(r80 >= 0.8, r80 < 15),
                      10.0**(B0 + B1*np.log10(r80)
                                + B2*np.log10(r80)**2
                                + B3*np.log10(r80)**3
                                + B4*np.log10(r80)**4), dFdr80)
    dFdr80 = np.where(np.logical_and(r80 >= 15, r80 < 37.5), C1*r80**-1, dFdr80)
    dFdr80 = np.where(np.logical_and(r80 >= 37.5, r80 < 100), C2*r80**-2.8, dFdr80)
    dFdr80 = np.where(np.logical_and(r80 >= 100, r80 < 250), C3*r80**-8, dFdr80)
    dFdr0_11ms = dFdr80*0.506*r0**-0.024    # A92 source function at 11 m/s, based on r0 [m-2 s-1 (\mu m)-1]
    
    if Wform == 'MOM80':
        WC_A92_11ms = whitecap_MOM80(np.array([11.0]))    # Whitecap fraction of A92 source function at 11 m/s [-]
        WC = whitecap_MOM80(wspd10)    # Whitecap fraction for field of wspd10 values [-]
    elif Wform == 'CF21':
        WC_A92_11ms = whitecap_CF210504(np.array([11.0]))    # Whitecap fraction of A92 source function at 11 m/s [-]
        WC = whitecap_CF210504(wspd10)    # Whitecap fraction for field of wspd10 values [-]
    dFdr0_perWC = dFdr0_11ms/WC_A92_11ms*1e6    # F94 number source function per unit whitecap area [m-2 s-1 m-1]
    dVdr0_perWC = dFdr0_perWC*4/3*np.pi*r0**3    # F94 volume source function per unit whitecap area [m3 m-2 s-1 m-1]
    dmdr0 = fs*rho_w*np.array([d*WC for d in dVdr0_perWC])    # F94 mass source function [kg m-2 s-1 m-1]

    # Calculate total spray mass flux
    dims = np.shape(wspd10)
    M_spr = np.zeros_like(wspd10)
    for i in range(dims[0]):
        for j in range(dims[1]):
            M_spr[i,j] = np.dot(dmdr0[:,i,j],delta_r0)    # [kg m-2 s-1]

    return r0,delta_r0,M_spr,dmdr0
        
def whitecapActive_DLM17(dcp,ustar,swh):
    """
    Active breaking whitecap fraction per Deike, Lenain, and Melville 2017.  This is a 
    parameterization based on volume of entrained air.
    Parameters:
        dcp - dominant phase velocity [m s-1]
        ustar - friction velocity [m s-1]
        swh - significant wave height [m]
    Return:
        Wa - actively breaking whitecap fraction [-]
    """
    g = 9.81    # Acceleration due to gravity [m s-2]
    Wa = 0.018*dcp*ustar**2/g/swh    # Active breaking whitecap fraction [-]
    Wa[Wa > 1] = 1    # Cap at 1.0
    return Wa

def whitecap_CF210504(wspd10):
    """
    Wind-based whitecap fraction received from Chris Fairall on May 4, 2021.
    Parameters:
        wspd10 - 10m windspeed [m s-1]
    Return:
        W - whitecap fraction [-]
    """
    W = 6.5e-4*np.maximum(wspd10 - 2.,0.)**1.5    # Whitecap fraction [-]
    W[W > 1] = 1    # Cap at 1.0
    return W

def whitecap_MOM80(wspd10):
    """
    Whitecap fraction per Monahan and O'Muircheartaigh 1980.
    Parameters:
        wspd10 - 10m windspeed [m s-1]
    Return:
        W - whitecap fraction [-]
    """
    W = 3.8e-6*wspd10**3.4    # Whitecap fraction [-]
    W[W > 1] = 1    # Cap at 1.0
    return W
