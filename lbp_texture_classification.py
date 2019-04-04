
import skimage.filters as filters

def getLbpFeatures(greyScaleImage):



    lbp = filters.local_binary_pattern(greyScaleImage, P=0,R=3, method='uniform')