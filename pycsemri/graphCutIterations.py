import ctypes
import numpy as np
import sys
from importlib import resources



def graphCutIterations(imDataParams, algoParams, residual, lmap, cur_ind):

    residual_p = np.ascontiguousarray(residual, dtype=np.float64)
    lmap_p = np.ascontiguousarray(lmap, dtype=np.float64)
    cur_ind_p = np.ascontiguousarray(cur_ind, dtype=np.float64)
    images = np.ascontiguousarray(imDataParams['images'], dtype=np.complex128)
    TE = np.ascontiguousarray(imDataParams['TE'], dtype=np.float64)
    fat_freq_list = algoParams['species'][1]['frequency']
    fat_freq = np.ascontiguousarray(fat_freq_list, dtype=np.float64)
    range_fm = np.ascontiguousarray(algoParams['range_fm'], dtype=np.float64)
    sx, sy, N, C, num_acqs, phase = imDataParams['images'].shape

    CPP_OUTPUT_DIR = './'

    # Load the shared library into ctypes
    lib_name = None
    if sys.platform.startswith('win'):
        lib_name = 'libgraphCutIterations.dll'
    elif sys.platform.startswith('linux'):
        lib_name = 'libgraphCutIterations.so'
    elif sys.platform.startswith('darwin'): # macOS
        lib_name = 'libgraphCutIterations.dylib'
    with resources.path('pycsemri', lib_name) as lib_path:
        lib = ctypes.CDLL(str(lib_path))


    lib.graphCutIterations_c_wrapper.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS'), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.POINTER(ctypes.POINTER(ctypes.c_bool)), ctypes.c_char_p]

    lib.graphCutIterations_c_wrapper.restype = None

    fm_out_ptr, mask_out_ptr = ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_bool)()

    lib.graphCutIterations_c_wrapper(sx, sy, num_acqs, len(TE), images, TE, imDataParams['FieldStrength'], 
                                     fat_freq, len(fat_freq_list), algoParams['lambda'], range_fm, algoParams['NUM_FMS'],
                                    algoParams['NUM_ITERS'], algoParams['size_clique'], residual_p, lmap_p, cur_ind_p, 
                                    ctypes.byref(fm_out_ptr), 
                                    ctypes.byref(mask_out_ptr), 
                                    CPP_OUTPUT_DIR.encode('utf-8'))

    array_size = sx * sy
    fm_array = np.ctypeslib.as_array(fm_out_ptr, shape=(array_size,))
    mask_array = np.ctypeslib.as_array(mask_out_ptr, shape=(array_size,))
    fm = np.copy(np.reshape(fm_array, (sx, sy)))
    masksignal = np.copy(np.reshape(mask_array, (sx, sy)))
    lib.free_memory_cpp(fm_out_ptr)
    lib.free_memory_cpp(mask_out_ptr)
    
    return fm, masksignal
        

