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
    }

    print ('Case,Cycle,Time,Ser,OFF_X,OFF_Y,OFF_Z')

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

        offsetDict = {}
        
        if freg == 3:
            param['initialOffset'] = [-imageData[7][0], -imageData[7][1], -imageData[7][2]]
        else:
            param['initialOffset'] = [0.0, 0.0, 0.0]

        if exam in planImageDict.keys():
            ablationImagePath = 'PC%03d/NIFTY-Iceball-AXTSE/%s' % (int(exam), ablationImageFile)
            ablationLabelPath = 'PC%03d/NIFTY-Iceball-AXTSE-label/%s' % (int(exam), ablationLabelFile)
            
            planImagePath = 'PC%03d/NRRD/%s' % (int(exam), planImageDict[exam][0])
            planStructureLabelPath = 'PC%03d/NIFTY-Anatomy-label/%s' % (int(exam), planImageDict[exam][1])
            
            ablationImage = sitk.ReadImage(ablationImagePath, sitk.sitkFloat32)
            ablationLabel = sitk.ReadImage(ablationLabelPath, sitk.sitkFloat32)              
            planImage = sitk.ReadImage(planImagePath, sitk.sitkFloat32)
            structureLabel = sitk.ReadImage(planStructureLabelPath, sitk.sitkUInt16)

            # Registration
            offset = [0.0, 0.0, 0.0]
            transform = None
            if freg > 0:
                movingImage = ablationImage
                fixedImage = planImage
                mask = None
                if freg == 2:
                    #movingImage = ar.mask(movingImage, structureLabel, dilation=10)
                    mask = ar.createMaskFromAnatomLabel(structureLabel, fixedImage, dilation=30)
                transform = ar.registerImages(fixedImage, movingImage, param, mask=mask, maskType='fixed')
                offset = transform.GetParameters()
            else:
                transform = sitk.Transform()

            outputImageDir = 'PC%03d/NIFTY-Iceball-Resampled' % int(exam)
            outputLabelDir = 'PC%03d/NIFTY-Iceball-Resampled-label' % int(exam)
            registeredImagePath = '%s/REG-ICEBALL-%s' % (outputImageDir, ablationImageFile)
            registeredLabelPath = '%s/REG-ICEBALL-%s' % (outputLabelDir, ablationLabelFile)
            ablationImageResampled = ar.resampleImage(ablationImage, planImage, transform, interp='linear')
            ablationLabelResampled = ar.resampleImage(ablationLabel, planImage, transform, interp='nearest')

            if not os.path.exists(outputImageDir):
                os.mkdir(outputImageDir)
              
            if not os.path.exists(outputLabelDir):
                os.mkdir(outputLabelDir)
              
            sitk.WriteImage(ablationImageResampled, registeredImagePath)
            sitk.WriteImage(ablationLabelResampled, registeredLabelPath)
            
            print ('%d,%d,%d,%d,%f,%f,%f' % (int(exam),
                                             fz,
                                             time,
                                             ser,
                                             offset[0],offset[1],offset[2])
                   )
            
        
if __name__ == "__main__":
  main(sys.argv[1:])



