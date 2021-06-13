import cv2
import numpy as np

from Airlight import Airlight
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze

if __name__ == '__main__':
    HazeImg = cv2.imread('../img/bench.jpg')

    # Resize image
    

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20
    C1 = 300        # Default value = 300
    # Computing the Transmission using equation
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)

    # Refine estimate of transmission
    # Default value = 1 --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    regularize_lambda = 1
    sigma = 0.5
    # Using contextual information
    Transmission = CalTransmission(
        HazeImg, Transmission, regularize_lambda, sigma)

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    cv2.imshow('Original', HazeImg)
    cv2.imshow('Result', HazeCorrectedImg)
    cv2.waitKey(0)

    #cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)
