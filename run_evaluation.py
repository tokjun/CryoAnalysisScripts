#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
import ablation_evaluation as ae
import ablation_registration as ar
import numpy as np
import csv


anatomDict = {
    1 : 'TG',
    2 : 'EUS',
    3 : 'NVB'
}



def erodeDilateLabelByDistance(label, distance):
    # If distance > 0, dilate the label
    # If distance < 0, erode the label
    distMapFilter = sitk.SignedDanielssonDistanceMapImageFilter()
    distMap = distMapFilter.Execute(label)
    newLabel = sitk.BinaryThreshold(distMap, -np.Inf, distance, 1, 0)
    
    return newLabel


def thresholdDurationMap(dmap, minDuration, maxDuration=np.Inf):
    
    newLabel = sitk.BinaryThreshold(dmap, minDuration, maxDuration, 1, 0)
    
    return newLabel



def main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Process images listed in the image list file. ")
        parser.add_argument('list', metavar='IMAGE_LIST', type=str, nargs=1,
                            help='A JSON file that lists planning and intraprocedural images.')
        parser.add_argument('output', metavar='RESULT_CSV', type=str, nargs=1,
                            help='A csv file to store the results')
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

    imageListFilePath = args.list[0]
    resultFilePath = args.output[0]
    
    imageList = None
    with open(imageListFilePath, "r") as read_file:
        imageList = json.load(read_file)

    planImageDict = imageList['PLAN_IMAGES']
    intraImageList = imageList['INTRA_IMAGES']

    param = {
        'margin'      : 0.0,
    }


    # Open the result csv file
    resultFile= open(resultFilePath, 'w', newline='')
    csvWriter = csv.writer(resultFile)

    resHeader = ['CASE', 'CYCLE', 'TIME', 'SERIES', 'MARGIN', 'MIN_DURATION', 'V_ABLATION', 'V_INV_TG', 'V_INV_EUS', 'V_INV_NVB']
    csvWriter.writerow(resHeader)
        

    # Calculate timestep
    intraImageMeta = []
    for line in intraImageList:
        intraImageMeta.append(line[0:4])
    intraImageMeta = np.array(intraImageMeta)
    intraImageMeta = np.concatenate((intraImageMeta, np.zeros((np.shape(intraImageMeta)[0],1))),axis=1)

    for exam in np.unique(intraImageMeta[:,0]):
        examMeta = intraImageMeta[intraImageMeta[:,0]==exam]
        for fz in np.unique(examMeta[:,1]):
            fzMeta = examMeta[examMeta[:,1]==fz]
            tShift = np.roll(fzMeta[:,2],1)
            tShift[0] = 0
            dt = fzMeta[:,2]-tShift
            #dt = dt.reshape([len(dt),1])
            intraImageMeta[(intraImageMeta[:,0]==exam) & (intraImageMeta[:,1]==fz),4] = dt
        
    print ('Case,Cycle,Time,Ser,Margin,V_TG,V_EUS,V_NVB,V_ablation,V_INV_TG,V_INV_EUS,V_INV_NVB')

    intraImageList2 = []
    for line in intraImageList:
        if len(line) < 8:
            line = line + [0.0, 0.0, 0.0]
        else:
            line = line[0:7] + line[7]
        intraImageList2.append(line)
    intraImageListNP = np.array(intraImageList2)

    margins = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0]

    for exam in np.unique(intraImageMeta[:,0]):
        examMeta = intraImageMeta[intraImageMeta[:,0]==exam]
        
        if not str(int(exam)) in planImageDict:
            continue
        
        planAnatomLabelPath = 'PC%03d/NIFTY-Anatomy-label/%s' % (int(exam), planImageDict[str(int(exam))][1])
        structureLabel = sitk.ReadImage(planAnatomLabelPath, sitk.sitkUInt16)
        
        refSize = structureLabel.GetSize()
        refOrigin = structureLabel.GetOrigin()
        refSpacing = structureLabel.GetSpacing()
        refDirection = structureLabel.GetDirection()

        durationMap = {}

        # Create maps
        for m in margins:
        
            dmap = sitk.Image(refSize, sitk.sitkFloat32)
            dmap.SetOrigin(refOrigin)
            dmap.SetSpacing(refSpacing)
            dmap.SetDirection(refDirection)

            durationMap[m] = dmap

            
        for fz in np.unique(examMeta[:,1]):
            fzMeta = examMeta[examMeta[:,1]==fz]
             
            for ser in fzMeta[:,3]:
                imageData = intraImageListNP[(intraImageListNP[:,0]==str(int(exam)))&(intraImageListNP[:,1]==str(int(fz)))&(intraImageListNP[:,3]==str(int(ser)))][0]
                time = float(imageData[2])
                #exam =     str(imageData[0])
                #fz   =     imageData[1]
                #time =     imageData[2]
                #ser  =     imageData[3]
                freg =     int(imageData[4])    # 0: No registration; 1: Registration w/o mask; 2: Registration w/ mask; 3: registration w/mask w/offset
                ablationLabelFile = imageData[6]
                
                dt = intraImageMeta[(intraImageMeta[:,0]==exam) & (intraImageMeta[:,1]==fz) & (intraImageMeta[:,3]==ser),4]
                
                if freg == 3:
                    param['initialOffset'] = [float(imageData[7]), float(imageData[8]), float(imageData[9])]
                
                if str(int(exam)) in planImageDict.keys():
                    ablationLabelPath = 'PC%03d/NIFTY-Iceball-Resampled-label/REG-ICEBALL-%s' % (int(exam), ablationLabelFile)
                      
                    ablationLabel = sitk.ReadImage(ablationLabelPath, sitk.sitkUInt16)

                    for m in margins:
                        label = None
                        if m == 0.0:
                            label = ablationLabel
                        else:
                            label = erodeDilateLabelByDistance(ablationLabel, m)

                        # Update map
                        durationMap[m] = sitk.Cast(label, sitk.sitkFloat32)*dt + durationMap[m]

                        results = ae.evaluateAblation(structureLabel, label, param, fMinDist=False)
                    
                        print ('%d,%d,%d,%d,%f, %f,%f,%f, %f, %f,%f,%f' %
                               (int(exam),
                                int(fz),
                                int(time),
                                int(ser),
                                m,
                                results['Structure.TG'],
                                results['Structure.EUS'],
                                results['Structure.NVB'],
                                results['AblationVolume'],
                                results['Involved.TG'],
                                results['Involved.EUS'],
                                results['Involved.NVB']))
                        
        
        # Output duration map
        mapDir = 'PC%03d/NIFTY-Map' % (int(exam))
        if not os.path.exists(mapDir):
            os.mkdir(mapDir)

        for m in margins:
            durationMapPath = '%s/DurationMap_%f.nii.gz' % (mapDir, m)
            sitk.WriteImage(durationMap[m], durationMapPath)

        # Compute summary
        minDurations = [0.001, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

        for m in margins:
            for d in minDurations:
                label = thresholdDurationMap(durationMap[m], d)
                results = ae.evaluateAblation(structureLabel, label, param, fMinDist=False)
                row = [str(int(exam)),
                       str(int(fz)),
                       str(int(time)),
                       str(int(ser)),
                       str(m),
                       str(d),
                       str(results['AblationVolume']),
                       str(results['Involved.TG']),
                       str(results['Involved.EUS']),
                       str(results['Involved.NVB'])]
                csvWriter.writerow(row)

    resultFile.close()
            
        
if __name__ == "__main__":
  main(sys.argv[1:])



