from pathlib import Path
import platform
import ctypes
import sys

if platform.system() == "Windows":
    buildFolder = Path("iq-neuron/out/build/x64-Debug")
    libiqPath = str(buildFolder / "iq-network.dll")
    libizPath = str(buildFolder / "iz-network.dll")
    liblifPath = str(buildFolder / "lif-network.dll")
else:
    buildFolder = Path("build/")
    libiqPath = buildFolder / "libiq-network.so"
    libizPath = buildFolder / "libiz-network.so"
    liblifPath = buildFolder / "liblif-network.so"

try:
    libiq = ctypes.CDLL(libiqPath)
except OSError:
    print("Unable to load iq shared library.")
    sys.exit()

try:
    libiz = ctypes.CDLL(libizPath)
except OSError:
    print("Unable to load iz shared library.")
    sys.exit()

try:
    liblif = ctypes.CDLL(liblifPath)
except OSError:
    print("Unable to load lif shared library.")
    #print("Unable to load iq/iz/lif shared library.")
    sys.exit()

print("All libs loaded. Congrats!")

class iqnet(object):
    def __init__(self, par, con):
        #libiq.iq_network_new.argtypes = None
        libiq.iq_network_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        libiq.iq_network_new.restype = ctypes.c_void_p

        libiq.iq_network_num_neurons.argtypes = [ctypes.c_void_p]
        libiq.iq_network_num_neurons.restype = ctypes.c_int

        libiq.iq_network_send_synapse.argtypes = [ctypes.c_void_p]
        libiq.iq_network_send_synapse.restype = ctypes.c_void_p

        libiq.iq_network_set_biascurrent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        libiq.iq_network_set_biascurrent.restype = ctypes.c_void_p

        libiq.iq_network_potential.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiq.iq_network_potential.restype = ctypes.c_int

        libiq.iq_network_spike_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiq.iq_network_spike_count.restype = ctypes.c_int

        libiq.iq_network_spike_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiq.iq_network_spike_rate.restype = ctypes.c_float

        libiq.iq_network_set_num_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiq.iq_network_set_num_threads.restype = ctypes.c_void_p

        b_par = par.encode('utf-8')
        b_con = con.encode('utf-8')
        self.obj = libiq.iq_network_new(b_par, b_con)

    def num_neurons(self):
        return libiq.iq_network_num_neurons(self.obj)

    def send_synapse(self):
        return libiq.iq_network_send_synapse(self.obj)

    def set_biascurrent(self, neuron_index, biascurrent):
        return libiq.iq_network_set_biascurrent(self.obj, neuron_index, biascurrent)

    def potential(self, neuron_index):
        return libiq.iq_network_potential(self.obj, neuron_index)

    def spike_count(self, neuron_index):
        return libiq.iq_network_spike_count(self.obj, neuron_index)

    def spike_rate(self, neuron_index):
        return libiq.iq_network_spike_rate(self.obj, neuron_index)

    def set_num_threads(self, num_threads):
        return libiq.iq_network_set_num_threads(self.obj, num_threads)

class iznet(object):
    def __init__(self):
        libiz.iz_network_new.argtypes = None
        libiz.iz_network_new.restype = ctypes.c_void_p

        libiz.iz_network_num_neurons.argtypes = [ctypes.c_void_p]
        libiz.iz_network_num_neurons.restype = ctypes.c_int

        libiz.iz_network_send_synapse.argtypes = [ctypes.c_void_p]
        libiz.iz_network_send_synapse.restype = ctypes.c_void_p

        libiz.iz_network_set_biascurrent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        libiz.iz_network_set_biascurrent.restype = ctypes.c_void_p

        libiz.iz_network_potential.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiz.iz_network_potential.restype = ctypes.c_float

        libiz.iz_network_adaptive_term.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiz.iz_network_adaptive_term.restype = ctypes.c_float

        libiz.iz_network_spike_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiz.iz_network_spike_count.restype = ctypes.c_int

        libiz.iz_network_spike_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libiz.iz_network_spike_rate.restype = ctypes.c_float

        self.obj = libiz.iz_network_new()

    def num_neurons(self):
        return libiz.iz_network_num_neurons(self.obj)

    def send_synapse(self):
        return libiz.iz_network_send_synapse(self.obj)

    def set_biascurrent(self, neuron_index, biascurrent):
        return libiz.iz_network_set_biascurrent(self.obj, neuron_index, biascurrent)

    def potential(self, neuron_index):
        return libiz.iz_network_potential(self.obj, neuron_index)

    def adaptive_term(self, neuron_index):
        return libiz.iz_network_adaptive_term(self.obj, neuron_index)

    def spike_count(self, neuron_index):
        return libiz.iz_network_spike_count(self.obj, neuron_index)

    def spike_rate(self, neuron_index):
        return libiz.iz_network_spike_rate(self.obj, neuron_index)

class lifnet(object):
    def __init__(self):
        liblif.lif_network_new.argtypes = None
        liblif.lif_network_new.restype = ctypes.c_void_p

        liblif.lif_network_num_neurons.argtypes = [ctypes.c_void_p]
        liblif.lif_network_num_neurons.restype = ctypes.c_int

        liblif.lif_network_send_synapse.argtypes = [ctypes.c_void_p]
        liblif.lif_network_send_synapse.restype = ctypes.c_void_p

        liblif.lif_network_set_biascurrent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        liblif.lif_network_set_biascurrent.restype = ctypes.c_void_p

        liblif.lif_network_potential.argtypes = [ctypes.c_void_p, ctypes.c_int]
        liblif.lif_network_potential.restype = ctypes.c_float

        liblif.lif_network_spike_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
        liblif.lif_network_spike_count.restype = ctypes.c_int

        liblif.lif_network_spike_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        liblif.lif_network_spike_rate.restype = ctypes.c_float

        self.obj = liblif.lif_network_new()

    def num_neurons(self):
        return liblif.lif_network_num_neurons(self.obj)

    def send_synapse(self):
        return liblif.lif_network_send_synapse(self.obj)

    def set_biascurrent(self, neuron_index, biascurrent):
        return liblif.lif_network_set_biascurrent(self.obj, neuron_index, biascurrent)

    def potential(self, neuron_index):
        return liblif.lif_network_potential(self.obj, neuron_index)

    def spike_count(self, neuron_index):
        return liblif.lif_network_spike_count(self.obj, neuron_index)

    def spike_rate(self, neuron_index):
        return liblif.lif_network_spike_rate(self.obj, neuron_index)

net = iqnet("inputs/neuronParameter_IQIF_bistable.txt", "inputs/Connection_Table_IQIF_bistable.txt")
