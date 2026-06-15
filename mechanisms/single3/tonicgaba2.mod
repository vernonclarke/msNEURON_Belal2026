TITLE gaba leak 

COMMENT
If want ohmic gaba use tonicgaba1
From Pavlov et al., 2009
Outwardly Rectifying Tonically Active GABAA Receptors in Pyramidal Cells Modulate Neuronal Offset, Not Gain 
ENDCOMMENT

NEURON{    SUFFIX gaba2
    NONSPECIFIC_CURRENT i    RANGE  i, e, gbar, minf}

UNITS {
	(mA)  = (milliamp)
	(mV)  =  (millivolt)
}

PARAMETER {    gbar = 0 (siemens/cm2)  }

ASSIGNED{    v (mV)    i (milliamp/cm2)    minf         e (mV)}

BREAKPOINT {    minf=a(v)/(a(v)+b(v))    i=gbar*minf*(v-e)}

FUNCTION a(v(mV)) {     LOCAL x          if (fabs(v+20) > 1e-5) {               x = 0.1 * (v + 20)         }else{              x = 0.1         }    a = (50 * x / (1 - exp(-x)))}

FUNCTION b(v(mV)) {    LOCAL x         if (fabs(v-10) > 1e-5) {              x = -0.08 * (v - 10)         } else {             x = -0.08         }    b = (20 * x / (1 - exp(-x)))}
