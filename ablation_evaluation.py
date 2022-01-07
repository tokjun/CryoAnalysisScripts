#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
#import sitkUtils

anatomDict = {
    1 : 'TG',
    2 : 'EUS',
    3 : 'NVB'
}


def computeDistanceFromAblationVolume(anatomLabel, ablationLabel):

    distMapFilter = sitk.SignedDanielssonDistanceMapImageFilter()
    distMap = distMapFilter.Execute(ablationLabel)
    
    labelStatistics = sitk.LabelStatisticsImageFilter()
    labelStatistics.Execute(distMap, anatomLabel)
    n = labelStatistics.GetNumberOfLabels()
    minDistances = {}
    for i in range(1,n):
        #print("%d," % i)                           #Index
        #print("%f," % labelStatistics.GetCount(i))  #Count
        #print("%f," % labelStatistics.GetMinimum(i))    #Min
        #print("%f," % labelStatistics.GetMaximum(i))    #Max
        #print("%f," % labelStatistics.GetMean(i))   #Mean
        #print("%f\n"% labelStatistics.GetSigma(i))  #StdDev
        minDistances[i] = float(labelStatistics.GetMinimum(i))

    return minDistances


def addMargin(srcLabel, margin):
    
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetBackgroundValue(0.0)
    dilate.SetKernelRadius(int(margin))
    dilate.SetForegroundValue(1.0)
    
    dstLabel = resampler.Execute(srcLabel)
    return dstLabel
    

def resampleImage(srcImage, refImage):
    
    dimension = srcImage.GetDimension()
    transform = sitk.AffineTransform(dimension)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(refImage)
    #resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)
    #resampler.SetOutputSpacing(RefImageSpacings)
    #resampler.SetSize(RefImage.GetSize())
    #resampler.SetOutputDirection(RefImage.GetDirection())
    resampler.SetOutputOrigin(refImage.GetOrigin())
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    resampler.SetDefaultPixelValue(0)
    
    dstImage = resampler.Execute(srcImage)
    return dstImage


def getLabelVolume(srcLabel):

    # Get the voxel volume
    dimension = srcLabel.GetDimension()
    spacing = srcLabel.GetSpacing()
    
    voxelVolume = 1.0
    for i in range(dimension):
        voxelVolume = voxelVolume * (spacing[i] / 10.0)

    #print('voxel spacing = %s' % str(spacing))
    #print('voxel volume = %f' % voxelVolume)
    labelStatistics = sitk.LabelStatisticsImageFilter()
    labelStatistics.Execute(srcLabel, srcLabel)
    n = labelStatistics.GetNumberOfLabels()
    volumes = {}
    
    for i in anatomDict:
        #print("%d," % i)                           #Index
        #print("%f," % labelStatistics.GetCount(i))  #Count
        #print("%f," % labelStatistics.GetMinimum(i))    #Min
        #print("%f," % labelStatistics.GetMaximum(i))    #Max
        #print("%f," % labelStatistics.GetMean(i))   #Mean
        #print("%f\n"% labelStatistics.GetSigma(i))  #StdDev
        volumes[i] = float(labelStatistics.GetCount(i)) * voxelVolume

    return volumes
        

def measureOverlap(srcLabel, maskLabel):
    
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(0.0)
    maskFilter.SetOutsideValue(1.0)
    overlapLabel = maskFilter.Execute(srcLabel, maskLabel)
    sitk.WriteImage(overlapLabel, 'overlap-label.nrrd')
    
    volumes = getLabelVolume(overlapLabel)
    return volumes


def evaluateAblation(structureLabel, ablationLabel, param):
    
    results = {
    }

    # Initialize
    for anatom in anatomDict.values():
        results['Structure.'+anatom] = 0.0
        results['Involved.'+anatom] = 0.0
    results['AblationVolume'] = 0.0        

    margin = param['margin']

    resampledStructureLabel = resampleImage(structureLabel, ablationLabel)
    
    structureVolumes = getLabelVolume(resampledStructureLabel)
    for key in structureVolumes:
        results['Structure.'+anatomDict[key]] = structureVolumes[key]
        
    
    if margin > 0.0:
        ablationLabel = addMargin(ablationLabel, margin)
        
    ablationVolumes = getLabelVolume(ablationLabel)
    if len(ablationVolumes) > 0:
        results['AblationVolume'] = ablationVolumes[1]

    overlapVolumes = measureOverlap(resampledStructureLabel, ablationLabel)
    for key in overlapVolumes:
        results['Involved.'+anatomDict[key]] = overlapVolumes[key]
        #print('Involved.'+anatomDict[key]+': '+str(overlapVolumes[key]))


    minDistances = computeDistanceFromAblationVolume(resampledStructureLabel, ablationLabel)
    for key in minDistances:
        results['MinDist.'+anatomDict[key]] = minDistances[key]
        #print('MinDist.'+anatomDict[key]+': '+str(minDistances[key]))

    return results


def ablation_evaluation_main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Map anatomy map onto intraop image by resampling.")
        parser.add_argument('plan', metavar='PLAN_LABEL_FILE', type=str, nargs=1,
                            help='A label map of critical structures.')
        parser.add_argument('intra', metavar='INTRA_LABEL_FILE', type=str, nargs=1,
                            help='A label map of ablatio volume.')
        #parser.add_argument('-c', dest='numberOfControlPoints', default='4,4,4',
        #                    help='Number of control points (default: 4,4,4)')
        parser.add_argument('-m', dest='ablationMargin', default='0.0',
                            help='Ablation Margin (default: 0.0)')
        #parser.add_argument('-b', dest='bSplineOrder', default='3',
        #                    help='B-Spline order (default: 3)')
        #parser.add_argument('-s', dest='shrinkFactor', default='4',
        #                    help='Shring factor (default: 4)')
        #parser.add_argument('-i', dest='numberOfIterations', default='50,40,30',
        #                    help='Number of iteration for each step (default: 50,40,30)')

        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    planFile = args.plan[0]
    intraFile = args.intra[0]
    param = {
        'margin': float(args.ablationMargin)
    }

    structureLabel = sitk.ReadImage(planFile, sitk.sitkUInt16)
    ablationLabel = sitk.ReadImage(intraFile, sitk.sitkUInt16)

    results = evaluateAblation(structureLabel, ablationLabel, param)
    
    for key in results:
        print('%s \t: %f cc' % (key, results[key]))
        
    #with open("data_file.json", "w") as write_file:
    #    json.dump([results, results], write_file)
    #
    #with open("image_list.json", "r") as read_file:
    #    data = json.load(read_file)
    #
    #for key in data[0]:
    #    print('%s \t: %s ' % (key, data[0][key]))
        
        
if __name__ == "__main__":
    ablation_evaluation_main(sys.argv[1:])


