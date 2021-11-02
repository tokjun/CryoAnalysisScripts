#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
import csv
import numpy as np


def computeMetrics(mesHeader, mesTable, volNamesAnatom, volNamesAblation):
    
    colTime       = mesHeader.index('Time')
    #colSer        = mesHeader.index('Ser')

    time1 = mesTable[:,colTime]
    time0 = np.roll(time1, 1)
    time0[0] = 0.0
    timeInterval = time1-time0

    metrics = {}
    
    for name in volNamesAblation:
        col = mesHeader.index(name)
        v = mesTable[:,col]
        v_max = np.max(v)
        v_dur = np.sum(timeInterval[v > 0.0])
        metrics[name+'_max'] = v_max
        metrics[name+'_duration'] = v_dur
        
    for name in volNamesAnatom:
        col = mesHeader.index(name)
        metrics[name] = mesTable[:,col][0]

    return metrics


def processMeasurementTable(mesHeader, mesTable, param):

    # Create a list of cases
    colCase = mesHeader.index('Case')
    colCycle = mesHeader.index('Cycle')
    caseList = np.unique(mesTable[:,colCase])

    volNamesAnatom = ['V_TG','V_EUS','V_NVB']
    volNamesAblation = ['V_ablation','V_INV_TG','V_INV_EUS','V_INV_NVB']

    resTable = []
    resHeader = ['Case', 'Cycle'] + volNamesAnatom
    for name in volNamesAblation:
        resHeader.append(name+'_max')
        resHeader.append(name+'_duration')
    
    for case in caseList:
        caseMesTable = mesTable[mesTable[:,colCase] == case, :]
        cycleList = np.unique(caseMesTable[:,colCycle])
        for cycle in cycleList:
            cycleMesTable = caseMesTable[caseMesTable[:,colCycle] == cycle, :]
            metrics = computeMetrics(mesHeader, cycleMesTable, volNamesAnatom, volNamesAblation)
            
            r = [int(case), int(cycle)]
            for name in volNamesAnatom:
                r.append(metrics[name])
                
            for name in volNamesAblation:
                r.append(metrics[name+'_max'])
                r.append(metrics[name+'_duration'])
            resTable.append(r)
        
    return (resHeader, resTable)
            

def loadMeasurements(mesFile, header=True):

    colList = []
    mesList = []

    with open(mesFile, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if header:
                for c in row:
                    colList.append(c)
                header = False
            else:
                r = []
                for c in row:
                    r.append(float(c))
                mesList.append(r)
                
    return (colList, np.array(mesList))


def outputMeasurements(resHeader, resTable, outFile):

    with open(outFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(resHeader)
        for row in resTable:
            srow = [str(s) for s in row]
            writer.writerow(srow)
    

def ablation_metrics_main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Compute ablation metrics")
        parser.add_argument('mes_csv', metavar='MEASUREMENT_FILE', type=str, nargs=1,
                            help='Measurement results in the CSV format. (Input)')
        parser.add_argument('out_csv', metavar='OUTPUT_FILE', type=str, nargs=1,
                            help='Computation results in the CSV format. (Ouput)')

        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    mesFile = args.mes_csv[0]
    outFile = args.out_csv[0]

    (mesHeader, mesTable) = loadMeasurements(mesFile)
    
    param = {}

    (resHeader, resTable) = processMeasurementTable(mesHeader, mesTable, param)
    
    outputMeasurements(resHeader, resTable, outFile)
        
        
if __name__ == "__main__":
    ablation_metrics_main(sys.argv[1:])



