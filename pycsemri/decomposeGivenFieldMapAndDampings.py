import ctypes
import numpy as np
import sys
from importlib import resources


def decomposeGivenFieldMapAndDampings( imDataParams,algoParams,fieldmap,r2starWater,r2starFat ):

    lib_name = None
    if sys.platform.startswith('win'):
        lib_name = 'libdecomposeGivenFieldMapAndDampings.dll'
    elif sys.platform.startswith('linux'):
        lib_name = 'libdecomposeGivenFieldMapAndDampings.so'
    elif sys.platform.startswith('darwin'): # macOS
        lib_name = 'libdecomposeGivenFieldMapAndDampings.dylib'
    with resources.path('pycsemri', lib_name) as lib_path:
        cpp_lib = ctypes.CDLL(str(lib_path))

    # Prepare inputs for C++ function
    c_double_p = ctypes.POINTER(ctypes.c_double)
    precessionIsClockwise = imDataParams.get('PrecessionIsClockwise', 1)
    ampW = algoParams['species'][0].get('relAmps', 1.0)
    gyro = 42.58
    deltaF = np.concatenate((np.array([0]), gyro * (algoParams['species'][1]['frequency'] - algoParams['species'][0]['frequency'][0]) * (imDataParams['FieldStrength']))).astype(np.float64)
    relAmps = np.array(algoParams['species'][1]['relAmps'], dtype=np.float64)
    t = np.array(imDataParams['TE'], dtype=np.float64)
    
    images = imDataParams['images']
    sx, sy, _, C, N, _ = images.shape
    images_squeezed = np.squeeze(images).astype(np.complex128)
    images_real = np.ascontiguousarray(images_squeezed.real)
    images_imag = np.ascontiguousarray(images_squeezed.imag)
    
    fieldmap_c = np.ascontiguousarray(fieldmap, dtype=np.float64)
    r2starWater_c = np.ascontiguousarray(r2starWater, dtype=np.float64)
    r2starFat_c = np.ascontiguousarray(r2starFat, dtype=np.float64)

    # Allocate output buffers
    amps_orig = np.zeros((sx, sy, 2, C), dtype=complex)
    amps_cpp_real = np.ascontiguousarray(np.zeros_like(amps_orig).real)
    amps_cpp_imag = np.ascontiguousarray(np.zeros_like(amps_orig).imag)

    # Allocate separate real/imaginary debug buffers
    debug_kx, debug_ky = sx // 2, sy // 2
    B1_cpp_real = np.zeros((N, 2), dtype=np.float64)
    B1_cpp_imag = np.zeros((N, 2), dtype=np.float64)
    B_cpp_real = np.zeros((N, 2), dtype=np.float64)
    B_cpp_imag = np.zeros((N, 2), dtype=np.float64)
    s_cpp_real = np.zeros((N, 1), dtype=np.float64)
    s_cpp_imag = np.zeros((N, 1), dtype=np.float64)


    cpp_func = cpp_lib.decomposeGivenFieldMapAndDampings_cpp
    cpp_func.restype = None
    cpp_func.argtypes = [
        c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, 
        c_double_p, c_double_p, # amps_out
        ctypes.c_int, ctypes.c_int, # debug kx, ky
        c_double_p, c_double_p, # B1_out
        c_double_p, c_double_p, # B_out
        c_double_p, c_double_p  # s_out
    ]
    
    #print("\nRunning C++ version...")
    cpp_func(
        images_real.ctypes.data_as(c_double_p), images_imag.ctypes.data_as(c_double_p),
        fieldmap_c.ctypes.data_as(c_double_p), r2starWater_c.ctypes.data_as(c_double_p), r2starFat_c.ctypes.data_as(c_double_p),
        t.ctypes.data_as(c_double_p), deltaF.ctypes.data_as(c_double_p), relAmps.ctypes.data_as(c_double_p),
        sx, sy, C, N, len(relAmps), ampW, precessionIsClockwise,
        amps_cpp_real.ctypes.data_as(c_double_p), amps_cpp_imag.ctypes.data_as(c_double_p),
        debug_kx, debug_ky,
        B1_cpp_real.ctypes.data_as(c_double_p), B1_cpp_imag.ctypes.data_as(c_double_p),
        B_cpp_real.ctypes.data_as(c_double_p), B_cpp_imag.ctypes.data_as(c_double_p),
        s_cpp_real.ctypes.data_as(c_double_p), s_cpp_imag.ctypes.data_as(c_double_p)
    )
    #print("...Done.")

    # Reconstruct complex data from C++ buffers
    amps_cpp = amps_cpp_real + 1j * amps_cpp_imag

    return amps_cpp

