#!/usr/bin/python3

################################################################################
# Generates, checks and packs submission for the challenge.

from os import getcwd
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory
from time import time
from zipfile import ZipFile, ZIP_DEFLATED

_, dpCodePath, inDataFilename, outDataFilename, inDataSpecsFilename, numColumns = sys.argv

cwd = getcwd()
tmpdir = TemporaryDirectory()

# Generates synthetic datasets with epsilon equal 1.0, 0.1, and 0.01.

file1 = Path(tmpdir.name, '1_0.csv')
file2 = Path(tmpdir.name, '0_1.csv')
file3 = Path(tmpdir.name, '0_01.csv')

run([Path(cwd, dpCodePath), inDataFilename, file1, inDataSpecsFilename, str(1.0), str(numColumns)])
run([Path(cwd, dpCodePath), inDataFilename, file2, inDataSpecsFilename, str(0.1), str(numColumns)])
run([Path(cwd, dpCodePath), inDataFilename, file3, inDataSpecsFilename, str(0.01), str(numColumns)])

# Checks dataset size limits

if file1.stat().st_size / 1024 / 1024 > 100:
  print('Error: Dataset generated for epsilon = 1.0 is larger than 100Mb')
  sys.exit(1)
if file2.stat().st_size / 1024 / 1024 > 100:
  print('Error: Dataset generated for epsilon = 0.1 is larger than 100Mb')
  sys.exit(1)
if file3.stat().st_size / 1024 / 1024 > 100:
  print('Error: Dataset generated for epsilon = 0.01 is larger than 100Mb')
  sys.exit(1)

zipf = ZipFile(Path(cwd, outDataFilename), 'w', ZIP_DEFLATED)
zipf.write(file1, '1_0.csv')
zipf.write(file2, '0_1.csv')
zipf.write(file3, '0_01.csv')
zipf.close()

print('Done')
