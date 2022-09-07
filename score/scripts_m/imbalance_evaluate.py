#!/usr/bin/python3

################################################################################
# This script generates ground truth data out of the pre-processed dataset and
# its meta-inromation object.

import csv
import json
import sys
from random import randint
from time import time

# Number of buckets for counting of numeric values (integers and floats)
NUM_BUCKETS = 100

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 10000


def generate_imbalance_set3(inDataFilename, inDataSpecsFilename, outGroundTruth):
    # Opens necessary files.
    print("Processing ground 1")
    inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
    inDataSpecs = json.load(open(inDataSpecsFilename))
    dataHeader = next(inData)

    # Randomly selects up to 100 sets of 3 columns each.
    columnSets = [];
    MAX_COLUMN_ID = len(dataHeader) - 1;
    while len(columnSets) < 100:
        col3 = randint(0, MAX_COLUMN_ID - 1);
        col2 = randint(0, MAX_COLUMN_ID - 1);
        col1 = randint(0, MAX_COLUMN_ID - 1);
        if (col3 != col2 and col3 != col1 and col2 != col1):
            columnSets.append([col1, col2, col3])

    ################################################################################
    # Collecting stats.

    print('Collecting stats...')

    counts = {}


    def countRow(row, bucket, cols, depth):
        value = row[cols[depth]]
        d = inDataSpecs[dataHeader[cols[depth]]]
        if d['type'] == 'enum':
            value = float(value)
            bucketId = int(value)
        elif value == '':
            bucketId = ''
            # print('blank@!')
            # exit()
        else:  # integer or float
            bucketSize = 1.0001 * float(d['max'] - d['min']) / NUM_BUCKETS
            bucketId = int((float(value) - d['min']) / bucketSize)

        if depth == 2:  # We are at leaf > just increment the counter.
            if bucketId in bucket:
                bucket[bucketId] += 1
            else:
                bucket[bucketId] = 1
            # if(bucketId==''):
            #   print(bucket[''])
            #   exit()
        else:
            if bucketId not in bucket: bucket[bucketId] = {}
            countRow(row, bucket[bucketId], cols, 1 + depth)


    rowIndex = 0
    for row in inData:
        # Logging
        rowIndex += 1
        if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
            print(f'{rowIndex / 1000 :12}k rows processed')
        # Count the row for each ground truth chunk.
        for index in range(len(columnSets)):
            if index not in counts: counts[index] = {}
            countRow(row, counts[index], columnSets[index], 0)

    ################################################################################
    # Output.

    print('Output...')

    outGT = csv.writer(open(outGroundTruth, 'w', newline=''), dialect='excel')

    for testId in counts:
        test = counts[testId]
        outGT.writerow(['@NEWCASE!'] + columnSets[testId])
        for key1 in test:
            for key2 in test[key1]:
                max_val = 0.0
                min_val = 99999999.0
                max_key = -1
                min_key = -1
                n = len(test[key1][key2])
                count = 0
                for key3 in test[key1][key2]:
                    value = test[key1][key2][key3]
                    count += value
                    if max_val < value:
                        max_val = value
                        max_key = key3
                    if min_val > value:
                        min_val = value
                        min_key = key3
                IR = max_val/min_val
                p_min = min_val/count
                if IR >= 10:
                    outGT.writerow([key1, key2, min_key, max_key, IR, p_min, 1/float(n)])


def generate_imbalance_set2(inDataFilename, inDataSpecsFilename, outGroundTruth):
    # Opens necessary files.
    print("Processing ground 1")
    inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
    inDataSpecs = json.load(open(inDataSpecsFilename))
    dataHeader = next(inData)

    # Randomly selects up to 100 sets of 3 columns each.
    columnSets = [];
    MAX_COLUMN_ID = len(dataHeader) - 1;
    while len(columnSets) < 100:
        col2 = randint(0, MAX_COLUMN_ID - 1);
        col1 = randint(0, MAX_COLUMN_ID - 1);
        if (col2 != col1):
            columnSets.append([col1, col2])

    ################################################################################
    # Collecting stats.

    print('Collecting stats...')

    counts = {}


    def countRow(row, bucket, cols, depth):
        value = row[cols[depth]]
        d = inDataSpecs[dataHeader[cols[depth]]]
        if d['type'] == 'enum':
            value = float(value)
            bucketId = int(value)
        elif value == '':
            bucketId = ''
            # print('blank@!')
            # exit()
        else:  # integer or float
            bucketSize = 1.0001 * float(d['max'] - d['min']) / NUM_BUCKETS
            bucketId = int((float(value) - d['min']) / bucketSize)

        if depth == 1:  # We are at leaf > just increment the counter.
            if bucketId in bucket:
                bucket[bucketId] += 1
            else:
                bucket[bucketId] = 1
            # if(bucketId==''):
            #   print(bucket[''])
            #   exit()
        else:
            if bucketId not in bucket: bucket[bucketId] = {}
            countRow(row, bucket[bucketId], cols, 1 + depth)


    rowIndex = 0
    for row in inData:
        # Logging
        rowIndex += 1
        if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
            print(f'{rowIndex / 1000 :12}k rows processed')
        # Count the row for each ground truth chunk.
        for index in range(len(columnSets)):
            if index not in counts: counts[index] = {}
            countRow(row, counts[index], columnSets[index], 0)

    ################################################################################
    # Output.

    print('Output...')

    outGT = csv.writer(open(outGroundTruth, 'w', newline=''), dialect='excel')

    for testId in counts:
        test = counts[testId]
        outGT.writerow(['@NEWCASE!'] + columnSets[testId])
        for key1 in test:
            max_val = 0.0
            min_val = 99999999.0
            max_key = -1
            min_key = -1
            n = len(test[key1])
            count = 0
            for key2 in test[key1]:
                value = test[key1][key2]
                count += value
                if max_val < value:
                    max_val = value
                    max_key = key2
                if min_val > value:
                    min_val = value
                    min_key = key2
            IR = max_val / min_val
            p_min = min_val/count

            if IR >= 10:
                outGT.writerow([key1, min_key, max_key, IR, p_min, 1/float(n)])




def imbalance_check_set3(inDataFilename, inGroundTruth1, inDataSpecsFilename, outBalance):
    print('*********************************************************************')
    print('SCORING METHOD #1')

    ##############################################################################
    # Initialization.
    f = csv.writer(open(outBalance, 'w', newline=''), dialect='excel')
    NUM_BUCKETS = 100

    print('Intitialization...')

    # Opens files.

    inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
    inGT = csv.reader(open(inGroundTruth1, newline=''), dialect='excel')
    inDataSpecs = json.load(open(inDataSpecsFilename))
    dataHeader = next(inData)

    # Loads ground truth data

    counts = {}
    columnSets = []

    for row in inGT:
        if row[0] == '@NEWCASE!':
            testId = len(columnSets)
            counts[testId] = {}
            columnSets.append([int(row[1]), int(row[2]), int(row[3])])
        else:
            test = counts[testId]
            key1 = row[0]
            key2 = row[1]
            key3 = row[2]
            max_key = row[3]
            IR = row[4]
            pmin = row[5]
            n = row[6]
            if key1 != '': key1 = int(key1)
            if key2 != '': key2 = int(key2)
            if key3 != '': key3 = int(key3)
            if max_key != '': max_key = int(max_key)
            if IR != '': IR = float(IR)
            if pmin != '': pmin = float(pmin)
            if n != '': n = float(n)
            if key1 not in test: test[key1] = {}
            if key2 not in test[key1]: test[key1][key2] = {}
            if key3 not in test[key1][key2]: test[key1][key2][key3] = [max_key, IR, pmin, n]

    ##############################################################################
    # Getting stats from the submission.

    print('Collecting stats from the solution...')

    solution = {}

    def countRow(row, bucket, cols, depth):
        value = row[cols[depth]]
        d = inDataSpecs[dataHeader[cols[depth]]]
        if d['type'] == 'enum':
            if value == '':
                bucketId = 'outrange'
            else:
                bucketId = int(float(value))
                if bucketId < 0 or bucketId >= d['count']: bucketId = 'outrange'
        elif value == '':
            bucketId = ''
        else:  # integer or float
            bucketSize = 1.0001 * float(d['max'] - d['min']) / NUM_BUCKETS
            bucketId = int((float(value) - d['min']) / bucketSize)
            if bucketId < 0 or bucketId >= NUM_BUCKETS: bucketId = 'outrange'

        # bucketId = 'outrange'

        if depth == 2:  # We are at leaf > just increment the counter.
            if bucketId in bucket:
                bucket[bucketId] += 1
            else:
                bucket[bucketId] = 1
        else:
            if bucketId not in bucket: bucket[bucketId] = {}
            bucket[bucketId] = countRow(row, bucket[bucketId], cols, 1 + depth)

        return bucket

    rowIndex = 0
    for row in inData:
        # Logging
        rowIndex += 1
        if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
            print(f'{rowIndex / 1000 :12}k rows processed')

        # if rowIndex == 10000: break

        # Count the row for each ground truth chunk
        for index in range(len(columnSets)):
            if index not in solution: solution[index] = {}
            cset = columnSets[index]
            if cset[2] < len(row) and cset[1] < len(row) and cset[0] < len(row):
                solution[index] = countRow(row, solution[index], cset, 0)
            else:
                solution[index] = {'outrange': {'outrange': {'outrange': rowIndex}}}

    ##############################################################################
    # Scoring.

    print('Scoring...')

    score = 0

    for testId in counts:
        if testId not in solution:
            score -= 1.0
        else:
            f.writerow([testId, columnSets[testId][0], columnSets[testId][1], columnSets[testId][2]])
            for key1 in solution[testId]:
                for key2 in solution[testId][key1]:
                    for key3 in solution[testId][key1][key2]:
                        v1 = float(solution[testId][key1][key2][key3])/rowIndex
                        if key1 in counts[testId] and key2 in counts[testId][key1] and key3 in counts[testId][key1][
                            key2]:
                            list_counts = counts[testId][key1][key2][key3]
                            f.writerow([key1, key2, key3, list_counts[0], list_counts[1], list_counts[2], list_counts[3],v1])
                        else:
                            score += 0



if __name__ == '__main__':
    startedAt = time()
    ################################################################################
    # Initialization.

    print('Initialization...')
    # Reads command-line arguments:
    # - Input dataset;
    # - Dictionary (meta-information) on the input dataset;
    # - Ground truth;
    _, in_DataFilename, in_DataSpecsFilename, out_GroundTruth_set3, out_GroundTruth_set2, inFile, out_balance = sys.argv
    generate_imbalance_set3(in_DataFilename, in_DataSpecsFilename, out_GroundTruth_set3)
    generate_imbalance_set2(in_DataFilename, in_DataSpecsFilename, out_GroundTruth_set2)

    imbalance_check_set3(inFile, out_GroundTruth_set3, in_DataSpecsFilename,out_balance)
    print(f'Time spent {(time() - startedAt) / 60} min')