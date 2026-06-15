'''
The MSN class defining the cell
'''

from neuron import h
import numpy as np
import json

# slight adjust to kas as slight aberrant distribution at VERY close dists to soma

# Distributions:
'''
T-type Ca: g = 1.0/( 1 +np.exp{(x-70)/-4.5} )
naf (den): (0.1 + 0.9/(1 + np.exp((x-60.0)/10.0)))

'''

def calculate_distribution(d3, dist, a4, a5,  a6,  a7, g8):
    '''
    Used for setting the maximal conductance of a segment.
    Scales the maximal conductance based on somatic distance and distribution type.

    Parameters:
    d3   = distribution type:
         0 linear, 
         1 sigmoidal, 
         2 exponential
         3 step function
    dist = somatic distance of segment
    a4-7 = distribution parameters 
    g8   = base conductance (similar to maximal conductance)

    '''

    if   d3 == 0: 
        value = a4 + a5*dist
    elif d3 == 1: 
        value = a4 + a5/(1 + np.exp((dist-a6)/a7) )
    elif d3 == 2: 
        value = a4 + a5*np.exp((dist-a6)/a7)
    elif d3 == 3:
        if (dist > a6) and (dist < a7):
            value = a4
        else:
            value = a5

    if value < 0:
        value = 0

    value = value*g8
    return value 


# ======================= the MSN class ==================================================
# compared to MSN_build1, removing Im and replace with KCNQ/Kv7
class MSN:
    def __init__(self,  params=None,      
                        morphology=None, 
                        variables=None,   
                        section=None,
                        replace_axon=False,
                        freq=2000,
                        d_lambda=0.05):
        Import = h.Import3d_SWC_read()
        Import.input(morphology)
        imprt = h.Import3d_GUI(Import, 0)
        imprt.instantiate(None)
        h.define_shape()
        # h.cao0_ca_ion = 2  # default in nrn
        h.celsius = 35
        self._create_sectionlists(replace_axon=replace_axon)
        self._set_nsegs2(freq=freq, d_lambda=d_lambda)
#         self._set_nsegs2(freq=2000, d_lambda=0.00670628)
#         self._set_nsegs(section=section)
        self.v_init = -84

        self.dendritic_channels =   [
                    "naf",      
                    "kaf",
                    "kas",
                    "kdr",
                    "kir",
                    "cal12",
                    "cal13",
                    "can",
                    "car",
                    "cav32",
                    "cav33",
                    "kcnq",
                    "sk",
                    "bk",
                    "gaba1",
                    "gaba2"       ]

        self.somatic_channels = [
                    "naf",
                    "kaf",
                    "kas",
                    "kdr",
                    "kir",
                    "cal12",
                    "cal13",
                    "cav32",
                    "cav33",
                    "kcnq",
                    "can",
                    "car",
                    "sk",
                    "nap",
                    "bk",
                    "gaba1",
                    "gaba2"     ]

        self.axonal_channels = [
                    "naf",
                    "kas" ,
                    "kcnq"    ]


        # insert active mechanisms (related to channels) -------------
        for sec in self.somalist:
            for mech in self.somatic_channels+["cadyn", "caldyn"]:
                sec.insert(mech)

        for sec in self.axonlist:
            for mech in self.axonal_channels:
                sec.insert(mech)

        for sec in self.dendlist:
            for mech in self.dendritic_channels+["cadyn", "caldyn"]:
                sec.insert(mech)

        with open(params) as file:
            par = json.load(file)

        # set passive parameters --------------------------------------------        
        for sec in self.allseclist:
            if not sec.name().find('spine') >= 0: # some mechs don't exist at spine, when remaking cell causes error
                sec.Ra = 200
                sec.cm = 1.0
                sec.insert('pas')
                #sec.g_pas = 1e-5 # set using json file
                sec.e_pas = float(par['e_pas']['Value']) 
                sec.g_pas = float(par['g_pas_all']['Value'])
                sec.ena = 50
                sec.ek = float(par['ek']['Value']) 

        self.distribute_channels("soma", "gbar_naf",   0, 1, 0, 0, 0, float(par['gbar_naf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_nap",   0, 1, 0, 0, 0, float(par['gbar_nap_somatic' ]['Value']))
        
        self.distribute_channels("soma", "gbar_kaf",   0, 1, 0, 0, 0, float(par['gbar_kaf_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kas",   0, 1, 0, 0, 0, float(par['gbar_kas_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kdr",   0, 1, 0, 0, 0, float(par['gbar_kdr_somatic']['Value']))
        self.distribute_channels("soma", "gbar_kir",   0, 1, 0, 0, 0, float(par['gbar_kir_somatic']['Value'])) 
        self.distribute_channels("soma", "gbar_bk",    0, 1, 0, 0, 0, float(par['gbar_bk_somatic']['Value']))
        self.distribute_channels("soma", "gbar_sk",    0, 1, 0, 0, 0, float(par['gbar_sk_basal']['Value']))
         
        self.distribute_channels("soma", "gbar_kcnq",  0, 1, 0, 0, 0, float(par['gbar_kcnq_somatic']['Value']))
        
        self.distribute_channels("soma", "pbar_cal12", 0, 1, 0, 0, 0, float(par['pbar_cal12_somatic']['Value']))
        self.distribute_channels("soma", "pbar_cal13", 0, 1, 0, 0, 0, float(par['pbar_cal13_somatic']['Value']))
        self.distribute_channels("soma", "pbar_cav32", 0, 1, 0, 0, 0, 0) # off
        self.distribute_channels("soma", "pbar_cav33", 0, 1, 0, 0, 0, 0) # off
        self.distribute_channels("soma", "pbar_car",   0, 1, 0, 0, 0, float(par['pbar_car_somatic' ]['Value']))
        self.distribute_channels("soma", "pbar_can",   0, 1, 0, 0, 0, float(par['pbar_can_somatic' ]['Value']))
                
        # use gaba1 for ohmic extrasynaptic GABA; use gaba2 for outward rectifying GABA
        # default both off; if employed, choose 1
        self.distribute_channels("soma", "gbar_gaba1", 0, 1, 0, 0, 0, float(par['gbar_gaba']['Value']))
        self.distribute_channels("soma", "gbar_gaba2", 0, 1, 0, 0, 0, float(par['gbar_gaba']['Value']))
        
        self.distribute_channels("dend", "gbar_gaba1", 0, 1, 0, 0, 0, float(par['gbar_gaba']['Value']))
        self.distribute_channels("dend", "gbar_gaba2", 0, 1, 0, 0, 0, float(par['gbar_gaba']['Value']))
        
        self.distribute_channels("dend", "gbar_naf",   1, 0, 1, 50.0, 10.0, float(par['gbar_naf_basal']['Value']))
        
        # self.distribute_channels("dend", "gbar_kaf",   1, 0.5, 0.25, 120.0, 30.0, float(par['gbar_kaf_basal']['Value'])) 
        self.distribute_channels("dend", "gbar_kaf",   1,   1,  0.5, 120.0,  -30.0, float(par['gbar_kaf_basal']['Value']))
        
        # self.distribute_channels("dend", "gbar_kas",   2, 0.25, 5, 0.0, -10.0, float(par['gbar_kas_basal']['Value']))
        self.distribute_channels("dend", "gbar_kas",   2, 0.25, 2, 0.0, -10.0, float(par['gbar_kas_basal']['Value']))
        # self.distribute_channels("dend", "gbar_kas",   2,    1, 9, 0.0, -5.0,  float(par['gbar_kas_basal']['Value']))
        
        self.distribute_channels("dend", "gbar_kdr",   1, 0.25, 1, 50.0, 30.0, float(par['gbar_kdr_basal']['Value'])) 
        self.distribute_channels("dend", "gbar_kir",   0, 1, 0, 0, 0, float(par['gbar_kir_basal']['Value'])) 
        self.distribute_channels("dend", "gbar_bk",    0, 1, 0, 0, 0, float(par['gbar_bk_basal']['Value']))
        self.distribute_channels("dend", "gbar_sk",    0, 1, 0, 0, 0, float(par['gbar_sk_basal']['Value']))
        
        self.distribute_channels("dend", "gbar_kcnq",  1, 0.25, 1, 50.0, 30.0, float(par['gbar_kcnq_basal']['Value']))         

        self.distribute_channels("dend", "pbar_cal12", 0, 1, 0, 0, 0, float(par['pbar_cal12_dend']['Value']))
        self.distribute_channels("dend", "pbar_cal13", 0, 1, 0, 0, 0, float(par['pbar_cal13_dend']['Value']))
        self.distribute_channels("dend", "pbar_cav32", 1, 0, 1.0, 120.0, -30.0, float(par['pbar_cav32_dend']['Value']))
        self.distribute_channels("dend", "pbar_cav33", 1, 0, 1.0, 120.0, -30.0, float(par['pbar_cav33_dend']['Value']))
        self.distribute_channels("dend", "pbar_car",   0, 1, 0, 0, 0, float(par['pbar_car_basal' ]['Value'])) 
        self.distribute_channels("dend", "pbar_can",   0, 1, 0, 0, 0, float(par['pbar_can_dend' ]['Value'])) 

        self.distribute_channels("axon", "gbar_naf",   3, 1, 1.1, 30, 500, float(par['gbar_naf_axonal']['Value']))
        self.distribute_channels("axon", "gbar_kas",   0, 1, 0, 0, 0, float(par['gbar_kas_axonal']['Value']))
        self.distribute_channels("axon", "gbar_kcnq",  0, 1, 0, 0, 0, float(par['gbar_kcnq_axonal']['Value']))
        # self.distribute_channels("axon", "gbar_Im",    0, 1, 0, 0, 0, float(par['gbar_Im_axonal']['Value']))

    def _create_sectionlists(self, replace_axon=False):
        # soma
        self.nsomasec = 0
        self.somalist = h.SectionList()
        for sec in h.allsec():
            if sec.name().find('soma') >= 0:
                self.somalist.append(sec=sec)
                if self.nsomasec == 0:
                    self.soma = sec
                self.nsomasec += 1
        # dendrite
        self.dendlist = h.SectionList()
        for sec in h.allsec():
            if sec.name().find('dend') >= 0:
                self.dendlist.append(sec=sec)
        # axon
        self.axonlist = h.SectionList()
        if replace_axon:
            self._create_AIS()
        else:
            axon=[]
            for sec in h.allsec():
                if sec.name().find('axon') >= 0:
                    self.axonlist.append(sec=sec)
        # all
        self.allsecnames = []
        self.allseclist  = h.SectionList()
        for sec in h.allsec():
            self.allsecnames.append(sec.name())
            self.allseclist.append(sec=sec)


    def _set_nsegs(self, section=None, N=20):
        """ def seg/sec. if section: set seg ~= 1/um  """
        if section:
            dend_name = 'dend[' + str(int(section)) + ']'
            for sec in self.allseclist:
                if sec.name() == dend_name:
                    n = 10*int(sec.L/2.0)+1
                    if n > N:
                        sec.nseg = n
                    else:
                        sec.nseg = 10*(N/2) + 1 # odd number of seg
                else:
                    sec.nseg = 10*int(sec.L/40.0)+1
        else:
            for sec in self.allseclist:
                sec.nseg = 10*int(sec.L/40.0)+1
        for sec in self.axonlist:
            sec.nseg = 20  # two segments in axon initial segment

    def _set_nsegs2(self, freq=100, d_lambda=0.1):
        """ def seg/sec """
        for sec in h.allsec():
            # apply lambda rule (Hines & Carnevale, 2001)
            AC_length = h.lambda_f(freq, sec = sec)
            sec.nseg = int((sec.L/(d_lambda*AC_length)+0.999)/2)*2 + 1
            
        self.soma.nseg = 1

    # def _set_nsegs2(self, freq=100, d_lambda=0.1):
    #     for sec in h.allsec():
    #         AC_length = h.lambda_f(freq, sec=sec)
    #         sec.nseg = int((sec.L / (d_lambda * AC_length) + 0.999) / 2) * 2 + 1
    #     self.soma.nseg = 1
    #     h.define_shape()

    def distribute_channels(self, as1, as2, d3, a4, a5, a6, a7, g8):
        h.distance(0, 0.5, sec=self.soma)

        for sec in self.allseclist:

            # if right cellular compartment (axon, soma or dend)
            if sec.name().find(as1) >= 0:
                for seg in sec:
                    loc = seg.x
                    dist = h.distance(seg.x, sec=sec)
                    val = calculate_distribution(d3, dist, a4, a5, a6, a7, g8)
                    cmd = 'seg.%s = %g' % (as2, val)
                    exec(cmd)



    def _create_AIS(self):
        """Replica of "Replace axon" in: 
            https://bluepyopt.readthedocs.io/en/latest/_modules/bluepyopt/ephys/morphologies.html#Morphology
        """

        temp = []
        for sec in h.allsec():
            if sec.name().find('axon') >= 0:
                temp.append(sec)

        # specify diameter based on blu
        if len(temp) == 0:
            ais_diams = [1, 1]
        elif len(temp) == 1:
            ais_diams = [temp[0].diam, temp[0].diam]
        else:
            ais_diams = [temp[0].diam, temp[0].diam]
            # Define origin of distance function
            h.distance(0, 0.5, sec=self.soma)

            for section in h.allsec():
                if section.name().find('axon') >= 0:
                    # If distance to soma is larger than 60, store diameter
                    if h.distance(1, 0.5, sec=section) > 60:
                        ais_diams[1] = section.diam
                        break

        # delete old axon
        for section in temp:
            h.delete_section(sec=section)

        # Create new axon sections
        a0 = h.Section(name='axon[0]')
        a1 = h.Section(name='axon[1]')

        # populate axonlist
        for sec in h.allsec():
            if sec.name().find('axon') >= 0:
                self.axonlist.append(sec=sec)

        # connect axon sections to soma and each other
        a0.connect(self.soma)
        a1.connect(a0)

        # set axon params
        for index, section in enumerate([a0,a1]):
            section.nseg = 1
            section.L = 30
            section.diam = ais_diams[index]

        # this line is needed to prevent garbage collection of axon 
        self.axon = [a0,a1]

        logger.debug('Replace axon with AIS') 

class Spine():
    """
    Spine class. Create a spine with neck and head.
    inspired by Mattioni and Le Novere, (2013).
    https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=150284&file=/TimeScales-master/neuronControl/spine.py#tabs-2

    if a parent section is passed as argument, the spine will be connected to that section (using the default orientation)
        to connect a spine at a later stage use the connect_spine(parent) method, where parent is a section.

    Since default orientation is used, to place multiple spines spread over one section, first use function split_section(sec) in common_functions.
        split_section(sec) splits one section into multiple sections with retained total surface area (and close conductances).
    """

    def __init__(self, id,              \
                       parent=None,     \
                       x=None,          \
                       neck_L=1.0,      \
                       neck_dia=0.1,    \
                       head_L=0.5,      \
                       head_dia=0.5,    \
                       Ra=200.0,        \
                       Cm=1.0,
                       g_pas=1.25e-5,
                       e_pas=-86,
                                ):
        """ Create a spine with geometry given by the arguments"""

        self.id         =   id
        self.neck       =   self.create_neck(neck_L=neck_L, neck_dia=neck_dia, Ra=Ra, Cm=Cm, g_pas=g_pas, e_pas=e_pas)
        self.head       =   self.create_head(head_L=head_L, head_dia=head_dia, Ra=Ra, Cm=Cm, g_pas=g_pas, e_pas=e_pas)
        self.parent     =   parent  # the parent section connected to the neck
        self.stim       =   None    # attribute for saving spike apparatus (netStim, synapse and
        self.x          =   x

        # connect spine parts
        self.connect_head2neck()

        # connect spine to parent
        self.connect_spine(parent, x)



    def create_neck(self, neck_L=1.0, neck_dia=0.1, Ra=200.0, Cm=1.0, g_pas=1.25e-5, e_pas=-86):
        """ Create a spine neck"""

        sec_name        =   'spine_%d_neck' % (self.id)
        neck            =   h.Section(name=sec_name)
        neck.nseg       =   1
        neck.L          =   neck_L 
        neck.diam       =   neck_dia
        neck.Ra         =   Ra 
        neck.cm         =   Cm

        for mech in [   'pas',      \
                        'cav32',    \
                        'cav33',    \
                        'caldyn'     ]:
            neck.insert(mech)

        neck(0.5).pbar_cav33 = 1e-7
        neck(0.5).pbar_cav33 = 1e-8


        neck.g_pas      =  g_pas
        neck.e_pas      =  e_pas 

        return neck

    def create_head(self, head_L=0.5, head_dia=0.5, Ra=200.0, Cm=1.0, g_pas=1.25e-5, e_pas=-86):
        """Create the head of the spine and populate it with channels"""

        sec_name        =   'spine_%d_head' % (self.id)
        head            =   h.Section(name=sec_name)

        head.nseg       =   1
        head.L          =   head_L
        head.diam       =   head_dia
        head.Ra         =   Ra
        head.cm         =   1.0

        for mech in [   'pas',      \
                        'kir',      \
                        'cav32',    \
                        'cav33',    \
                        'car',      \
                        'cal12',    \
                        'cal13',    \
                        'cadyn',    \
                        'sk',       \
                        'caldyn'    ]:
            head.insert(mech)

        head(0.5).pbar_cav32 = 1e-7
        head(0.5).pbar_cav33 = 1e-8
        head(0.5).pbar_car   = 1e-8
        head(0.5).pbar_cal12 = 1e-5 # 1e-7
        head(0.5).pbar_cal13 = 1e-6 # 1e-8
        head(0.5).gbar_kir   = 1e-7
        head(0.5).gbar_sk    = 0.00002

        head.g_pas      =   g_pas
        head.e_pas      =   e_pas 

        return head

    def connect_head2neck(self, parent=None):
        ''' connect spine head to neck and if parent is not None connect spine to parent 
        connection is hardcoded to use default connection orientation (0 end connected to 1 end on parent)
        To use other orientation, first create and then connect spine, 
            using the connect spine method with updated arguments
        '''
        self.head.connect(self.neck(1),0)
        if not parent is None:
            self.neck.connect(parent(1),0) 

    def connect_spine(self, parent, x=1, end=0):
        ''' connect spine to parent sec.
        '''
        self.neck.connect(parent(x),end)
        self.parent = parent

    def move_spine(self, new_parent):
        ''' move spine from one section to another (using default orientation)
        '''
        h.disconnect(sec=self.neck)
        self.neck.connect(new_parent(1),0)
        self.parent = new_parent
                   
#     replace int(sec.L * spine_per_length / sec.nseg) with max(1, round(sec.L * spine_per_length / sec.nseg))
#     replace nspines = int((x-30)*b) with n_spines = max(1, round((x-30)*b))
#     this ensures that values less than 0.5 are rounded up to 1, not down to 0 and has a minimum spine count of 1 in a seg

def add_spines(cell, spine_per_length, verbose=True): 
    SPINES = {}
    ID = 0
    for sec in cell.dendlist:
        spines_per_seg = max(1, round(sec.L * spine_per_length / sec.nseg)) # int(sec.L * spine_per_length / sec.nseg)
        SPINES[sec.name()] = {}
        for i,seg in enumerate(sec):
            x = h.distance(seg) # distance to soma(0.5)
            if 80 > x > 30: # none 0-30, 30+ linear increase
                b = spines_per_seg / (80-30)
                n_spines = max(1, round((x-30)*b)) # round((x-30)*b) # int((x-30)*b)
                for j in range(n_spines):
                    SPINES[sec.name()][ID] = Spine(ID,sec,seg.x + (j + 0.5 * (1 - n_spines))/n_spines/sec.nseg)
                    # ensures all spines have unique distance from soma
                    # seg.x + (j + 0.5 * (1 - n_spines))/n_spines/sec.nseg
                    # in reality they will be lumped into seg.x
                    # SPINES[sec.name()][ID] = Spine(ID,sec,seg.x)
                    ID += 1
                    
            elif x >= 80: # peak constant 
                n_spines = spines_per_seg
                for j in range(n_spines):
                    SPINES[sec.name()][ID] = Spine(ID,sec,seg.x + (j + 0.5 * (1 - n_spines))/n_spines/sec.nseg)
                    ID += 1
                    
    if verbose:
        print("number of spines added:{}.".format(ID))
    return SPINES
