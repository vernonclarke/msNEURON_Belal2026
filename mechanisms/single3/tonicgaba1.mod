TITLE gaba leak 

COMMENT
If want outward rectification use tonicgaba2
ENDCOMMENT

NEURON{    SUFFIX gaba1
    NONSPECIFIC_CURRENT i    RANGE  i, e, gbar}

UNITS {
	(mA)  = (milliamp)
	(mV)  =  (millivolt)
}

PARAMETER {    gbar = 0 (siemens/cm2)   }

ASSIGNED{    v (mV)    i (milliamp/cm2)    minf         e (mV)}

BREAKPOINT {    i=gbar*(v-e)}

