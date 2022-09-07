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


def generate_ground_truth1(inDataFilename, inDataSpecsFilename, outGroundTruth):
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
                for key3 in test[key1][key2]:
                    value = float(test[key1][key2][key3]) / rowIndex
                    outGT.writerow([key1, key2, key3, value])


def generate_ground_truth1_set3(inDataFilename, inDataSpecsFilename, outGroundTruth):
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
                for key3 in test[key1][key2]:
                    value = float(test[key1][key2][key3]) / rowIndex
                    outGT.writerow([key1, key2, key3, value])


def generate_ground_truth1_set4(inDataFilename, inDataSpecsFilename, outGroundTruth):
    # Opens necessary files.
    print("Processing ground 1")
    inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
    inDataSpecs = json.load(open(inDataSpecsFilename))
    dataHeader = next(inData)

    # Randomly selects up to 100 sets of 3 columns each.
    columnSets = [];
    MAX_COLUMN_ID = len(dataHeader) - 1;
    while len(columnSets) < 100:
        col4 = randint(0, MAX_COLUMN_ID - 1)
        col3 = randint(0, MAX_COLUMN_ID - 1);
        col2 = randint(0, MAX_COLUMN_ID - 1);
        col1 = randint(0, MAX_COLUMN_ID - 1);
        if (col3 != col2 and col3 != col1 and col2 != col1
                and col4 != col3 and col4 != col2 and col4 != col1):
            columnSets.append([col1, col2, col3, col4])

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

        if depth == 3:  # We are at leaf > just increment the counter.
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
                for key3 in test[key1][key2]:
                    for key4 in test[key1][key2][key3]:
                        value = float(test[key1][key2][key3][key4]) / rowIndex
                        outGT.writerow([key1, key2, key3, key4, value])

def generate_ground_truth1_set2(inDataFilename, inDataSpecsFilename, outGroundTruth):
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
    total_minor = 0
    s_freq = 0
    for testId in counts:
        test = counts[testId]
        outGT.writerow(['@NEWCASE!'] + columnSets[testId])
        unique_values = 0
        for key1 in test:
            for key2 in test[key1]:
                unique_values += 1
        threshold = 1/ (100 * unique_values)
        number_minor = 0
        for key1 in test:
            for key2 in test[key1]:
                value = float(test[key1][key2]) / rowIndex
                if value < threshold:
                    outGT.writerow([key1, key2, value, threshold, unique_values])
                    number_minor += 1
        outGT.writerow(['@END'] + [number_minor])
        total_minor += number_minor
    outGT.writerow(['@EOF'] + [total_minor])
    return total_minor


def generate_ground_truth1_set1(inDataFilename, inDataSpecsFilename, outGroundTruth, choose_columns=100):
    # Opens necessary files.
    print("Processing ground 1 - set 1 column")
    inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
    inDataSpecs = json.load(open(inDataSpecsFilename))
    dataHeader = next(inData)

    # Randomly selects up to 100 sets of 3 columns each.
    columnSets = [];
    MAX_COLUMN_ID = len(dataHeader) - 1;
    while len(columnSets) < choose_columns:
        # col3 = randint(0, MAX_COLUMN_ID - 1);
        # col2 = randint(0, MAX_COLUMN_ID - 1);
        col1 = randint(0, MAX_COLUMN_ID - 1);
        # if (col3 != col2 and col3 != col1 and col2 != col1):
        columnSets.append([col1])

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

        if depth == 0:  # We are at leaf > just increment the counter.
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
            value = float(test[key1]) / rowIndex
            outGT.writerow([key1, value])




if __name__ == '__main__':
    startedAt = time()
    ################################################################################
    # Initialization.

    print('Initialization...')
    # Reads command-line arguments:
    # - Input dataset;
    # - Dictionary (meta-information) on the input dataset;
    # - Ground truth;
    _, in_DataFilename, in_DataSpecsFilename, out_GroundTruth = sys.argv
    generate_ground_truth1_set1(in_DataFilename, in_DataSpecsFilename, out_GroundTruth)
    print(f'Time spent {(time() - startedAt) / 60} min')