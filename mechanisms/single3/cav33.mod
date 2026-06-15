TITLE T-type calcium channel (Cav3.3)

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
    (molar) = (1/liter)
    (mM) = (millimolar)
    FARADAY = (faraday) (coulomb)
    R = (k-mole) (joule/degC)
}

NEURON {
    SUFFIX cav33
    USEION cal READ cali, calo WRITE ical VALENCE 2
    RANGE pbar, ical, mvhalf, hvhalf, a, p, perm, I
    RANGE sf
}

PARAMETER {
    pbar = 6.7e-6 (cm/s)
    mvhalf = -73.5 (mV)     : 73.5 +/- 1.3
    mslope =  -4.4 (mV)
    hvhalf = -73.4 (mV)     : 73.4 +/- 2.5
    hslope =   5.6 (mV)
    a      = 0.9
    p      = 2
    sf = 1              : scaling factor for conductance
}

ASSIGNED { 
    v (mV)
    ical (mA/cm2)
    ecal (mV)
    celsius (degC)
    cali (mM)
    calo (mM)
    minf
    hinf
    mtau (ms)
    htau (ms)
    htau2 (ms)
    htot (ms)
    perm
    I
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    perm = pbar*(m^p)*h
    ical = ghk(v, cali, calo)*perm*sf
    I    = ical
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

DERIVATIVE states { 
    rates(v)
    m' = (minf-m)/mtau
    h' = (hinf-h)/htot
}

PROCEDURE rates(v (mV)) {
    minf = 1.0/(1+exp((v-mvhalf)/mslope))
    hinf = 1.0/(1+exp((v-hvhalf)/hslope))
    mtau = 6.0/(1+exp((v+73.0)/14.0))+0.8/(1+exp( (v-5.0)/3.0))+0.7 
    htau = 5.5/(1 + exp(0.06*(v)))+7
    htau2 = 90*exp(-(v+68.0)/40.0)+10
    htot  = a*htau + (1-a)*htau2
}

FUNCTION ghk(v (mV), ci (mM), co (mM)) (.001 coul/cm3) {
    LOCAL z, eci, eco
    z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))
    if(z == 0) {
        z = z+1e-6
    }
    eco = co*(z)/(exp(z)-1)
    eci = ci*(-z)/(exp(-z)-1)
    ghk = (1e-3)*2*FARADAY*(eci-eco)
}

COMMENT 

Original data by Iftinca (2006), rat, 37 C

Genesis implementation (22 C) by Kai Du <kaidu828@gmail.com>, m*h, also corrected
for m^2*h.

NEURON implementation by Alexander Kozlov <akozlov@nada.kth.se>, smooth
fit of mtau and htau.

Revised model by Robert Lindroos 
-> 37 C (kinetics and infinity parameters; before room temp were used and adjusted by q-factor)
-> m2 used directly on reported values (m) without adjustment (see, cav32.mod).
   This gives good fit to experimental IV curve.
-> slow and fast inactivation combined as 0.9*fast + 0.1*slow

Rat Cav3.2 channels were isolated and transfection of human embryonic
kidney cells was performed [1].  Electrophysiological recordings were
done in 21 C.

NEURON model by Alexander Kozlov <akozlov@kth.se>. Kinetics of m3h
type was used [2-4]. Activation time constant was scaled up accordingly.

[1] Iftinca M, McKay BE, Snutch TP, McRory JE, Turner RW, Zamponi
GW (2006) Temperature dependence of T-type calcium channel
gating. Neuroscience 142(4):1031-42.

[2] Crunelli V, Toth TI, Cope DW, Blethyn K, Hughes SW (2005) The
'window' T-type calcium current in brain dynamics of different behavioural
states. J Physiol 562(Pt 1):121-9.

[3] Wolf JA, Moyer JT, Lazarewicz MT, Contreras D, Benoit-Marand M,
O'Donnell P, Finkel LH (2005) NMDA/AMPA ratio impacts state transitions
and entrainment to oscillations in a computational model of the nucleus
accumbens medium spiny projection neuron. J Neurosci 25(40):9080-95.

[4] Evans RC, Maniar YM, Blackwell KT (2013) Dynamic modulation of
spike timing-dependent calcium influx during corticostriatal upstates. J
Neurophysiol 110(7):1631-45.

ENDCOMMENT
