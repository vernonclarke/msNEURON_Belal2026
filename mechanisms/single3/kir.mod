TITLE Inwardly rectifying potassium current

COMMENT
Neuromodulation is added as functions:
    
    modulationDA = 1 + modDA*(maxModDA-1)*levelDA

where:
    
    modDA  [0]: is a switch for turning modulation on or off {1/0}
    maxModDA [1]: is the maximum modulation for this specific channel (read from the param file)
                    e.g. 10% increase would correspond to a factor of 1.1 (100% +10%) {0-inf}
    levelDA  [0]: is an additional parameter for scaling modulation. 
                Can be used simulate non static modulation by gradually changing the value from 0 to 1 {0-1}
                                    
      Further neuromodulators can be added by for example:
      modulationDA = 1 + modDA*(maxModDA-1)
      modulationACh = 1 + modACh*(maxModACh-1)
      ....

      etc. for other neuromodulators
      
       
                                     
[] == default values
{} == ranges
    
ENDCOMMENT

NEURON {
    THREADSAFE
    SUFFIX kir
    USEION k READ ek WRITE ik
    RANGE gbar, gk, ik
    RANGE damod, maxMod, level, max2, lev2
    RANGE sf
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar        = 0.0    (S/cm2) 
    q           = 3
    modDA       = 0
    maxModDA    = 1
    levelDA     = 0
    modACh      = 0
    maxModACh   = 1
    levelACh    = 0
    mVhalf      = -102   (mV)   : -82    (mV)
    mSlope      =  13    (mV)   
    a           = 0.055  (/ms)  : 0.123  (/ms)
    alpha_half  = -60    (mV)   : -90.2  (mV)
    alpha_slope =  14    (mV)   :  65.1  (mV)
    b           = 0.125  (/ms)  : 0.196  (/ms)
    beta_half   = -31    (mV)   : -2.5   (mV)
    beta_slope  = -23    (mV)   : -13.2  (mV)
    sf          = 1              : scaling factor for conductance

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
    gk = gbar*m*modulationDA()*modulationACh()
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
    minf  = 1/(1+exp((v-mVhalf)/mSlope))
    alpha = a*exp(-(v-alpha_half)/alpha_slope)
    beta  = b/(1+exp((v-beta_half)/beta_slope))
    sum   = alpha+beta
    mtau  = 1/sum
    UNITSON
}

FUNCTION modulationDA() {
    : returns modulation factor
    
    modulationDA = 1 + modDA*(maxModDA-1)*levelDA 
}

FUNCTION modulationACh() {
    : returns modulation factor
    
    modulationACh = 1 + modACh*(maxModACh-1)*levelACh 
}


COMMENT

Original data by Steephen (2009), rat, room temp.

alpha and beta retuned to match raw data for mtau in paper: vernon.clarke@northwestern.edu

Genesis implementation by Kai Du <kai.du@ki.se>, MScell v9.5.

NEURON implementation by Alexander Kozlov <akozlov@csc.kth.se>, smooth
fit of mtau.

ENDCOMMENT
