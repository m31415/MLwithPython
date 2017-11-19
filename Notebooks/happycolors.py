import numpy as np

def gradientcolor(startClr_, endClr_, startVal_, endVal_, val_, tupleFlag_):
    """ This functions maps a start and endpoint with the given start 
    and end color in rgb (range 0 to 1 as tuple) and returns the corresponding 
    color of the given point within that range"""
    
    ## check inputs
    # check if only numerics are given
    assert not isinstance(startClr_, str)
    assert not isinstance(endClr_, str)
    assert isinstance(startVal_, (int, float))
    assert isinstance(endVal_, (int, float))
    assert isinstance(val_, (int, float))
    
    # check shapes
    assert (len(startClr_) == 3)
    assert (len(endClr_) == 3)
    
    # check if given value is within range of start and end value
    if (val_ > endVal_):
        valColor = endClr_
    elif (val_ < startVal_):
        valColor = startClr_
    else:
        # get value relative to given range
        relVal = (val_-startVal_)/(endVal_-startVal_)

        # create output color variable
        valColor = []
        
        # iterate over r g and b
        for element in range(3):
            # get distance between element of start/end color
            distance = np.abs(startClr_[element] - endClr_[element])
            
            if (startClr_[element] <= endClr_[element]):
                # compute element color 
                clr = (relVal * distance) + startClr_[element]
            else:
                clr = startClr_[element] -(relVal * distance)
            
            # append output
            valColor.append(clr)
    
    # should tuple be returned?
    if tupleFlag_:
        valColor = tuple(valColor)
    
    return valColor