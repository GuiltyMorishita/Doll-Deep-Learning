#! -*- coding: utf-8 -*-
import csv
import re

csvfile = "watanabe.csv"
f = open(csvfile, "r")
reader = csv.reader(f)

cnt = 0
answer = 0
for row in reader:
    matchOB = re.search(r'\\dataset\\dealer\\(\w+)\\', row[0])
    # matchOB = re.search(r'\\dataset\\brand\\(\w+)\\', row[0])
    if matchOB.group(1) == row[4]:
        answer += 1
    cnt += 1

    print(matchOB.group(1))
    print(row[4])

print(cnt, answer)
print(float(answer)/cnt)

f.close()
