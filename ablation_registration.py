#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
#import sitkUtils


def command_iteration(method):
    #print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")
    pass


def resampleImage(srcImage, refImage, transform, interp='linear'):
    
    dimension = srcImage.GetDimension()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(refImage)
    if interp == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(refImage.GetSpacing())
    resampler.SetSize(refImage.GetSize())
    resampler.SetOutputDirection(refImage.GetDirection())
    resampler.SetOutputOrigin(refImage.GetOrigin())
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    resampler.SetDefaultPixelValue(0)
    
    dstImage = resampler.Execute(srcImage)
    
    return dstImage


def mask(image, anatomLabel, dilation=0):
    
    relabelMap =  { i : 0 for i in range(2, 256) }
    maskLabel = sitk.ChangeLabel(anatomLabel, changeMap=relabelMap)

    if dilation > 0:
        filter = sitk.BinaryDilateImageFilter()
        filter.SetKernelRadius ( dilation )
        filter.SetForegroundValue ( 1 )
        maskLabel = filter.Execute ( maskLabel )
        
    #sitk.WriteImage(maskLabel, 'masklabel.nrrd')
    
    # Make sure that the mask and the moving image occupy the same physical space.
    transform = sitk.AffineTransform(3)
    maskLabel = resampleImage(maskLabel, image, transform, 'nearest')
    
    maskedImage = sitk.Mask(image, maskLabel, maskingValue=0, outsideValue=1)

    return maskedImage
    

def registerImages(fixedImage, movingImage, param):

    if not 'numberOfBins' in param:
        param['numberOfBins'] = 50
    if not 'samplingPercentage' in param:
        param['samplingPercentage'] = 0.1
        
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(param['numberOfBins'])
    R.SetMetricSamplingPercentage(param['samplingPercentage'], sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    #R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetInitialTransform(sitk.TranslationTransform(fixedImage.GetDimension()))
    #R.SetInitialTransform(sitk.Euler3DTransform())
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    transform = R.Execute(fixedImage, movingImage)

    # print("-------")
    # print(transform)
    # print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {R.GetOptimizerIteration()}")
    # print(f" Metric value: {R.GetMetricValue()}")


    return transform
    
    

def registration_main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Register two images .")
        parser.add_argument('fixed', metavar='FIXED_IMAGE', type=str, nargs=1,
                            help='Fixed image.')
        parser.add_argument('moving', metavar='MOVING_IMAGE', type=str, nargs=1,
                            help='Moving image')
        parser.add_argument('resampled', metavar='RESAMPLED_IMAGE', type=str, nargs=1,
                            help='Resampled')
        parser.add_argument('-m', dest='anatomLabel', default='',
                            help='Anatom label map for masking.')

        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    fixedImageFile = args.fixed[0]
    movingImageFile = args.moving[0]
    resampledImageFile = args.resampled[0]
    
    param = {
    }

    fixedImage = sitk.ReadImage (fixedImageFile, sitk.sitkFloat32)
    movingImage = sitk.ReadImage(movingImageFile, sitk.sitkFloat32)

    movingImageMasked = movingImage
    if args.anatomLabel != '':
        anatomLabel = sitk.ReadImage(args.anatomLabel, sitk.sitkInt8)
        movingImageMasked = mask(movingImage, anatomLabel, dilation=20)
        #sitk.WriteImage(movingImageMasked, 'maskedimage.nrrd')

    transform = registerImages(fixedImage, movingImageMasked, param)
    resampledImage = resampleImage(movingImage, fixedImage, transform)
    
    sitk.WriteImage(resampledImage, resampledImageFile)
    

if __name__ == "__main__":
    registration_main(sys.argv[1:])


