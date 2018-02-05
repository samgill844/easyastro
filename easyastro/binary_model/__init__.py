import numpy as np
import ctypes
from ctypes import *
import sys, os

current_dir =  os.path.dirname(os.path.realpath(__file__))



def make_nd_array(c_pointer, shape, dtype=np.float, order='C', own_data=True):
    arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize 
    if sys.version_info.major >= 3:
        buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        buffer = buf_from_mem(c_pointer, arr_size, 0x100)
    else:
        buf_from_mem = ctypes.pythonapi.PyBuffer_FromMemory
        buf_from_mem.restype = ctypes.py_object
        buffer = buf_from_mem(c_pointer, arr_size)
    arr = np.ndarray(tuple(shape[:]), dtype, buffer, order=order)
    if own_data and not arr.flags.owndata:
        return arr.copy()
        return arr
    else:
        return arr


# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_vector_functions():
    dll = ctypes.CDLL(current_dir + '/easy_binary.so', mode=ctypes.RTLD_GLOBAL)
	
    # RV function
    RV_function = dll.radvel
    RV_function.argtypes = [POINTER(c_float), c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, POINTER(c_float), POINTER(c_float), c_int]

    lc_function = dll.cuda_lc
    lc_function.argtypes = [POINTER(c_float), c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_int,POINTER(c_float), c_float,  POINTER(c_float), c_int,ctypes.c_char_p]

    return RV_function, lc_function




# create __cuda_sum function with get_cuda_sum()
__RV_function, __lc_function = get_vector_functions()


# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_rv(t, t0=0, p=1, v0=10, dv0=0, k1=20, k2=20, f_c=0.1, f_s=0.1):
    t = t.astype('float32')
    RV1 = np.zeros(t.shape[0]).astype('float32')
    RV2 = np.zeros(t.shape[0]).astype('float32')

    n = t.shape[0]

    t_p = t.ctypes.data_as(POINTER(c_float))
    RV1_p = RV1.ctypes.data_as(POINTER(c_float))
    RV2_p = RV2.ctypes.data_as(POINTER(c_float))

    __RV_function(t_p, t0, p, v0, dv0, k1, k2, f_c, f_s, RV1_p, RV2_p, n )
    RV1 = make_nd_array(RV1_p, [n], dtype=np.float32, order='C', own_data=True)
    RV2 = make_nd_array(RV2_p, [n], dtype=np.float32, order='C', own_data=True)

    return RV1, RV2




def cuda_lc(t, t0=0, p=1, radius_1 = 0.2, k=0.2, b = 0.05, f_c = 0.0, f_s = 0.0, ld_law_1 = "lin", ldc_1 = [0.6], S = 0.0, CPUorGPU= b'CPU'):

    '''
    Calculte limb-darkening for a variety of laws e.t.c.
    [0] linear (Schwarzschild (1906, Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
    [1] Quadratic Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
    [2] Square-root (Díaz-Cordovés & Giménez, 1992, A&A, 259, 227) 
    [3] Logarithmic (Klinglesmith & Sobieski, 1970, AJ, 75, 175)
    [4] Exponential LD law (Claret & Hauschildt, 2003, A&A, 412, 241)
    [5] Sing three-parameter law (Sing et al., 2009, A&A, 505, 891)
    [6] Claret four-parameter law (Claret, 2000, A&A, 363, 1081)
    '''
    if ld_law_1 == "lin" : ld_law_1=0;
    if ld_law_1 == "quad" : ld_law_1=1;
    if ld_law_1 == "squar" : ld_law_1=2;
    if ld_law_1 == "log" : ld_law_1=3;
    if ld_law_1 == "exp" : ld_law_1=4;
    if ld_law_1 == "sing" : ld_law_1=5;
    if ld_law_1 == "claret" : ld_law_1=6;

    t = t.astype('float32')
    I = np.zeros(t.shape[0]).astype('float32')
    n = t.shape[0]

    t_p = t.ctypes.data_as(POINTER(c_float))
    I_p = I.ctypes.data_as(POINTER(c_float))
    ldc_1_p = np.array(ldc_1).ctypes.data_as(POINTER(c_float))

    __lc_function(t_p, t0, p, radius_1, k, b, f_c, f_s, ld_law_1, ldc_1_p, S,  I_p, n, CPUorGPU)
    I = make_nd_array(I_p, [n], dtype=np.float32, order='C', own_data=True)


    return I


