#!/usr/bin/python3

################################################################################
# Local scorer.

import csv
import json
import sys
from math import log, sqrt

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 10000

# Reads command-line arguments:
# - Solution
# - Data Specs
# - Ground truth #1
# - Ground truth #2



################################################################################
# SCORING METHOD #1

def calcScore1(inData, inGT,inDataSpecs):
    print('*********************************************************************')
    print('SCORING METHOD #1')

    ##############################################################################
    # Initialization.

    NUM_BUCKETS = 100

    print('Intitialization...')

    # Opens files.

    # inData = csv.reader(open(data_file, newline=''), dialect='excel')
    # inGT = csv.reader(open(ground_truth1, newline=''), dialect='excel')
    # inDataSpecs = json.load(open(spec_file))
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
            count = row[3]
            if key1 != '': key1 = int(key1)
            if key2 != '': key2 = int(key2)
            if key3 != '': key3 = int(key3)
            if count != '': float(count)
            if key1 not in test: test[key1] = {}
            if key2 not in test[key1]: test[key1][key2] = {}
            if key3 not in test[key1][key2]: test[key1][key2][key3] = count

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
                bucketId = int(value)
                if bucketId < 0 or bucketId > d['maxval']: bucketId = 'outrange'
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

    score = len(counts.keys())

    for testId in counts:
        if testId not in solution:
            score -= 1.0
        else:
            for key1 in solution[testId]:
                for key2 in solution[testId][key1]:
                    for key3 in solution[testId][key1][key2]:
                        v1 = float(solution[testId][key1][key2][key3]) / rowIndex
                        if key1 in counts[testId] and key2 in counts[testId][key1] and key3 in counts[testId][key1][
                            key2]:
                            v2 = float(counts[testId][key1][key2][key3])
                            score += v2
                        else:
                            v2 = 0
                        score -= abs(v2 - v1)

    score *= 1000000.0 / 2 / len(counts.keys())

    print(f'SCORE #1 = {score}')
    return score


################################################################################
# SCORING METHOD #2

def calcScore2(inData, inGT):
    print('*********************************************************************')
    print('SCORING METHOD #2')

    ##############################################################################
    # Initialization.

    # inData = csv.reader(open(data_file, newline=''), dialect='excel')
    # inGT = csv.reader(open(ground_truth2, newline=''), dialect='excel')
    # dataHeader = next(inData)

    # Loads ground truth data.

    testCases = []
    testCase = None
    for row in inGT:
        if row[0] == '@NEWCASE!':
            if testCase != None: testCases.append(testCase)
            testCase = [float(row[1])]
        else:
            item = [int(row[0]), row[1]]
            if row[1] == 'enum':
                for x in row[2:]:
                    item.append(int(x))
            else:
                if (row[2] == ''):
                    item.append('')
                else:
                    item.append(float(row[2]))
                    item.append(float(row[3]))
            testCase.append(item)
    testCases.append(testCase)

    # Collects statistics from the solution
    counts = []
    for testCase in testCases:
        counts.append(0)

    numRows = 0
    for row in inData:
        numRows += 1
        if numRows % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
            print(f'{numRows / 1000 :12}k rows processed')
        for testCaseId in range(len(testCases)):
            count = True
            for item in testCases[testCaseId][1:]:
                columnId = item[0]
                itemType = item[1]
                if columnId >= len(row):
                    pass
                elif itemType == 'enum':
                    if int(row[columnId]) not in item[2:]:
                        count = False
                        break
                else:
                    if item[2] == '':
                        if row[columnId] != '':
                            count = False
                            break
                    elif row[columnId] == '':
                        count = False
                        break
                    else:
                        d = abs(float(row[columnId]) - item[2])
                        if d > item[3]:
                            count = False
                            break
            if count is True: counts[testCaseId] += 1

    sum2 = 0
    for index in range(len(testCases)):
        x = log(max(counts[index] / numRows, 1e-6))
        gt = log(testCases[index][0])
        sum2 += (x - gt) * (x - gt)

    score = 1e6 * max(0, (1 + sqrt(sum2 / len(testCases)) / log(1e-3)))
    print(f'SCORE #2 = {score}')
    return score

def calculate_scores(data_file, ground_truth1, ground_truth2, spec_file):
    inData = csv.reader(open(data_file, newline=''), dialect='excel')
    inGT1 = csv.reader(open(ground_truth1, newline=''), dialect='excel')
    inGT2 = csv.reader(open(ground_truth2, newline=''), dialect='excel')
    inDataSpecs = json.load(open(spec_file))
    score1 = calcScore1(inData, inGT1, inDataSpecs)
    inData = csv.reader(open(data_file, newline=''), dialect='excel')
    score2 = calcScore2(inData, inGT2)
    return score1, score2
    pass


################################################################################
# Score aggregation.
if __name__ == '__main__':
    list_args = sys.argv
    if len(list_args) == 6:
      _, inDataFilename, inDataSpecsFilename, inGroundTruth1, inGroundTruth2, SR_FOLDER = list_args
    else:
      _, inDataFilename, inDataSpecsFilename, inGroundTruth1, inGroundTruth2 = list_args
      SR_FOLDER = "score_results/log_score.txt"
    score1, score2 = calculate_scores(inDataFilename, inGroundTruth1, inGroundTruth2, inDataSpecsFilename)
    # score3 = calcScore3()

    score = (score1 + score2) / 2

    print('OVERAL SCORE =', score)

    f = open(SR_FOLDER, "a")
    f.write(
        inDataFilename + ", " + inDataSpecsFilename + ", " + inGroundTruth1 + ", " + inGroundTruth2 + ", " + str(score1)
        + ", " + str(score2) + ", " + str(score) + "\n")
    f.close()
    print("Saved to %s" % SR_FOLDER)
