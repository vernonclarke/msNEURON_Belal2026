TITLE KCNQ2/3 potassium channel Kv7


NEURON {
    SUFFIX kcnq
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik
    RANGE sf
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.0 (S/cm2) 
    q = 3
    theta_m = -59.5 (mV)  
    k_m = 10.3 (mV)      
    mtau0 = 6.7 (ms)
    mtau1 = 100.0 (ms)
    phi_m = -61.0 (mV)
    sigma_m0 = 35.0 (mV)
    sigma_m1 = -25.0 (mV)
    damod = 0
    maxMod = 1
    level = 0
    max2 = 1
    lev2 = 0
    sf = 1              : scaling factor for conductance
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    gk (S/cm2)
    minf
    mtau (ms)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gk = gbar*m*m*m*m*modulation()
    ik = sf*gk*(v-ek)
}

DERIVATIVE states {
    rates()
    m' = (minf-m)/mtau*q
}

INITIAL {
    rates()
    m = minf
}

PROCEDURE rates() {
    LOCAL alpha, beta, sum
    UNITSOFF
    minf = 1.0 / (1.0 + exp((theta_m - v)/k_m))
    mtau = mtau0 + (mtau1 - mtau0)/(exp((phi_m - v)/sigma_m0) + exp((phi_m - v)/sigma_m1))
    UNITSON
}

FUNCTION modulation() {
    : returns modulation factor
    
    modulation = 1 + damod * ( (maxMod-1)*level + (max2-1)*lev2 ) 
    if (modulation < 0) {
        modulation = 0
    } 
}

COMMENT
 https://modeldb.science/267669?tab=2&file=STN-GPe/KCNQ.mod
 modeled by Gunay et al., 2008
 implemented in NEURON by Kitano, 2011
 (mho/cm2) same as S/cm2
 gbar = 0.001

 theta_m = -61.0 (mV) changed to -59.5 then to -54.5
 k_m = 19.5 (mV) changed to 10.5
 to match Shen 2005 for SPNs
 q to 1 raises rheobase
 added modulation
ENDCOMMENT

