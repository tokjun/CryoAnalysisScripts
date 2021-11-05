#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
import ablation_evaluation as ae
import ablation_registration as ar


anatomDict = {
    1 : 'TG',
    2 : 'EUS',
    3 : 'NVB'
}


def main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Process images listed in the image list file. ")
        parser.add_argument('list', metavar='IMAGE_LIST', type=str, nargs=1,
                            help='A JSON file that lists planning and intraprocedural images.')
        parser.add_argument('-i', dest='saveRegImage', action='store_const',
                            const=True, default=False,
                            help='Save registered images')
        parser.add_argument('-l', dest='saveRegLabel', action='store_const',
                            const=True, default=False,
                            help='Save registered labels')

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

    imageListFile = args.list[0]
    
    imageList = None
    with open(imageListFile, "r") as read_file:
        imageList = json.load(read_file)

    planImageDict = imageList['PLAN_IMAGES']
    intraImageList = imageList['INTRA_IMAGES']

    param = {
        'margin': 0.0,
        'saveRegImage': args.saveRegImage,
        'saveRegLabel': args.saveRegLabel
    }

    print ('Case,Cycle,Time,Ser,V_TG,V_EUS,V_NVB,V_ablation,V_INV_TG,V_INV_EUS,V_INV_NVB,OFF_X,OFF_Y,OFF_Z')
    
    for imageData in intraImageList:
        exam =     str(imageData[0])
        fz   =     imageData[1]
        time =     imageData[2]
        ser  =     imageData[3]
        freg =     imageData[4]    # 0: No registration; 1: Registration w/o mask; 2: Registration w/ mask; 3: registration w/mask w/offset
        ablationImageFile = imageData[5]
        ablationLabelFile = imageData[6]
        if freg == 3:
            param['initialOffset'] = imageData[7]
        
        if exam in planImageDict.keys():
            ablationImagePath = 'PC%03d/NIFTY-Iceball-AXTSE/%s' % (int(exam), ablationImageFile)
            planImagePath = 'PC%03d/NRRD/%s' % (int(exam), planImageDict[exam][0])
            
            ablationLabelPath = 'PC%03d/NIFTY-Iceball-AXTSE-label/%s' % (int(exam), ablationLabelFile)
            planAnatomLabelPath = 'PC%03d/NIFTY-Anatomy-label/%s' % (int(exam), planImageDict[exam][1])
            
            planImage = sitk.ReadImage(planImagePath, sitk.sitkFloat32)
            ablationImage = sitk.ReadImage(ablationImagePath, sitk.sitkFloat32)

            structureLabel = sitk.ReadImage(planAnatomLabelPath, sitk.sitkUInt16)
            ablationLabel = sitk.ReadImage(ablationLabelPath, sitk.sitkUInt16)

            # Registration
            offset = [0.0, 0.0, 0.0]
            structureLabelResampled = structureLabel
            
            if freg > 0:
                movingImage = planImage
                fixedImage = ablationImage
                mask = None
                if freg == 2:
                    #movingImage = ar.mask(movingImage, structureLabel, dilation=10)
                    mask = ar.createMaskFromAnatomLabel(structureLabel, movingImage, dilation=30)
                transform = ar.registerImages(fixedImage, movingImage, param, mask=mask)
                structureLabelResampled = ar.resampleImage(structureLabel, ablationLabel, transform, interp='nearest')
                if param['saveRegLabel']:
                    registeredLabelPath = 'PC%03d/NIFTY-Anatomy-label/REG-LABEL-%d-%s' % (int(exam), ser, planImageDict[exam][1])
                    sitk.WriteImage(structureLabelResampled, registeredLabelPath)
            
                if param['saveRegImage']:
                    registeredImagePath = 'PC%03d/NIFTY-Anatomy-label/REG-IMAGE-%d-%s' % (int(exam), ser, planImageDict[exam][0])
                    planImageResampled = ar.resampleImage(planImage, ablationImage, transform, interp='linear')
                    sitk.WriteImage(planImageResampled, registeredImagePath)
                
                offset = transform.GetParameters()
            
            results = ae.evaluateAblation(structureLabelResampled, ablationLabel, param)

            #results = ae.evaluateAblation(structureLabel, ablationLabel, param)
            
            print ('%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f' % (int(exam),
                                                                  fz,
                                                                  time,
                                                                  ser,
                                                                  results['Structure.TG'],
                                                                  results['Structure.EUS'],
                                                                  results['Structure.NVB'],
                                                                  results['AblationVolume'],
                                                                  results['Involved.TG'],
                                                                  results['Involved.EUS'],
                                                                  results['Involved.NVB'],
                                                                  offset[0],offset[1],offset[2])
                   )
            
        
if __name__ == "__main__":
  main(sys.argv[1:])



