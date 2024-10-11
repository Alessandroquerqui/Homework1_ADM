# Homework1_ADM

## This file is needed to explain the contents of all the other files in this repository. Ciao
#!/bin/python3

import math
import os
import random
import re
import sys

from datetime import datetime

#I find the difference in seconds within the timestamps
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    dt1 = datetime.strptime(t1, time_format)
    dt2 = datetime.strptime(t2, time_format)
    # now the difference within the datetimes
    delta = abs((dt1 - dt2).total_seconds())
    return str(int(delta))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input()) 
    for t_itr in range(t):
        t1 = input().strip()
        t2 = input().strip()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n') 

