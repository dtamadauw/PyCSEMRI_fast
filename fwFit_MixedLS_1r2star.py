import ctypes
import numpy as np


# Load the shared library into ctypes
libuwwfs = ctypes.CDLL("/home/dxt005/lib/libuwwfs.so")



class ImDataParams_str(ctypes.Structure):
    _fields_ = [("TE", ctypes.c_double * 32),
                ("nte", ctypes.c_int),
                ("FieldStrength", ctypes.c_double),
                ("PrecessionIsClockwise", ctypes.c_double),
                ("images_r", ctypes.POINTER(ctypes.c_double)),
                ("images_i", ctypes.POINTER(ctypes.c_double)),
                ("im_dim", ctypes.c_int * 2)]

class AlgoParams_str(ctypes.Structure):
    _fields_ = [("species_wat_amp", ctypes.c_double * 32),
                ("species_wat_freq", ctypes.c_double * 32),
                ("species_fat_amp", ctypes.c_double * 32),
                ("species_fat_freq", ctypes.c_double * 32),
                ("NUM_WAT_PEAKS", ctypes.c_int),
                ("NUM_FAT_PEAKS", ctypes.c_int),
                ("NUM_FMS", ctypes.c_int)]

class InitParams_str(ctypes.Structure):
    _fields_ = [("water_r_init", ctypes.POINTER(ctypes.c_double)),
                ("fat_r_init", ctypes.POINTER(ctypes.c_double)),
                ("water_i_init", ctypes.POINTER(ctypes.c_double)),
                ("fat_i_init", ctypes.POINTER(ctypes.c_double)),
                ("r2s_init", ctypes.POINTER(ctypes.c_double)),
                ("fm_init", ctypes.POINTER(ctypes.c_double)),
				("masksignal_init", ctypes.POINTER(ctypes.c_double))]

class OutParams_str(ctypes.Structure):
    _fields_ = [("r2starmap", ctypes.POINTER(ctypes.c_double)),
                ("fm", ctypes.POINTER(ctypes.c_double)),
                ("wat_r_amp", ctypes.POINTER(ctypes.c_double)),
                ("fat_r_amp", ctypes.POINTER(ctypes.c_double)),
				("wat_i_amp", ctypes.POINTER(ctypes.c_double)),
				("fat_i_amp", ctypes.POINTER(ctypes.c_double))]
    


def fwFit_MixedLS_1r2star(imDataParams, algoParams, initParams):

	nte = len(imDataParams['TE'])
	TE = np.zeros(32, dtype=np.float64)
	FieldStrength = imDataParams['FieldStrength']
	TE[0:nte] = imDataParams['TE']
	PrecessionIsClockwise = imDataParams['PrecessionIsClockwise']
	img_dim = imDataParams['images'].shape
	#images_r = np.zeros(img_dim[0]*img_dim[1])
	#images_i = np.zeros(img_dim[0]*img_dim[1])
	images_r = np.real(imDataParams['images']).flatten(order='F').astype(np.float64)
	images_i = np.imag(imDataParams['images']).flatten(order='F').astype(np.float64)
	im_dim = np.zeros(2).astype(np.int32)
	im_dim[0] = img_dim[0]
	im_dim[1] = img_dim[1]
     

	#For algoParams_c
	NUM_WAT_PEAKS = 1
	NUM_FAT_PEAKS = len(algoParams['species'][1]['frequency'])
	species_wat_amp = np.zeros(32, dtype=np.float64)
	species_fat_amp = np.zeros(32, dtype=np.float64)
	species_wat_freq = np.zeros(32, dtype=np.float64)
	species_fat_freq = np.zeros(32, dtype=np.float64)
	species_wat_amp[0:NUM_WAT_PEAKS] = algoParams['species'][0]['relAmps']
	species_fat_amp[0:NUM_FAT_PEAKS] = algoParams['species'][1]['relAmps']
	species_wat_freq[0:NUM_WAT_PEAKS] =  algoParams['species'][0]['frequency']
	species_fat_freq[0:NUM_FAT_PEAKS] =  algoParams['species'][1]['frequency']
	NUM_FMS = algoParams['NUM_FMS']
     
	#For initParams
	water_r_init = np.real(initParams['amp_w']).flatten(order='F').astype(np.float64)
	water_i_init = np.imag(initParams['amp_w']).flatten(order='F').astype(np.float64)
	fat_r_init = np.real(initParams['amp_f']).flatten(order='F').astype(np.float64)
	fat_i_init = np.imag(initParams['amp_f']).flatten(order='F').astype(np.float64)
	r2s_init = initParams['r2starmap'].flatten(order='F').astype(np.float64)
	fm_init = initParams['fm'].flatten(order='F').astype(np.float64)
	masksignal_init= initParams['masksignal'].flatten(order='F').astype(np.float64)

	#For outParams
	wat_r_amp = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	wat_i_amp = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	fat_r_amp = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	fat_i_amp = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	r2starmap = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	fm = np.zeros(img_dim[0]*img_dim[1], dtype=np.float64)
	

	#Allocate Memories
	imDataParams_c = ImDataParams_str((ctypes.c_double * len(TE))(*TE), 
								nte, 
								FieldStrength, 
								PrecessionIsClockwise, 
								images_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
								images_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
								(ctypes.c_int * len(im_dim))(*im_dim))
	algoParams_c = AlgoParams_str((ctypes.c_double * 32)(*species_wat_amp),
								(ctypes.c_double * 32)(*species_wat_freq),
								(ctypes.c_double * 32)(*species_fat_amp),
								(ctypes.c_double * 32)(*species_fat_freq),
								NUM_WAT_PEAKS, NUM_FAT_PEAKS, NUM_FMS)

	initParams_c = InitParams_str(water_r_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                fat_r_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
								water_i_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
								fat_i_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
								r2s_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
								fm_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
								masksignal_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    
    
	outParams_c = OutParams_str(r2starmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
							fm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
							wat_r_amp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
							fat_r_amp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
							wat_i_amp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
							fat_i_amp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

	
	libuwwfs.fwFit_MixedLS_1r2star_c.argtypes = [ctypes.POINTER(ImDataParams_str), ctypes.POINTER(AlgoParams_str), ctypes.POINTER(InitParams_str), ctypes.POINTER(OutParams_str)]
	libuwwfs.fwFit_MixedLS_1r2star_c.restype = None
	libuwwfs.fwFit_MixedLS_1r2star_c(ctypes.byref(imDataParams_c), ctypes.byref(algoParams_c), ctypes.byref(initParams_c), ctypes.byref(outParams_c))
	
	outParams = {
        'water_amp': np.reshape(wat_r_amp + 1j*wat_i_amp, (im_dim[0],im_dim[1])),
		'fat_amp':  np.reshape(fat_r_amp + 1j*fat_i_amp, (im_dim[0],im_dim[1])),
        'r2starmap':  np.reshape(r2starmap, (im_dim[0],im_dim[1])),
        'fm':  np.reshape(fm, (im_dim[0],im_dim[1]))
	}

	return outParams
