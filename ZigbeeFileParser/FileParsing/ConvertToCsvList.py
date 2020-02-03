import time
from datetime import datetime, date, time, timedelta
timewindow = 5
first = 0
node_list = []
node_discrete_list = []
f = open("No_ms\\output4", "r")

for x in f:
    line = f.readline()
    split = line.split()
    if first <= 0:
        delta = datetime.strptime(split[0], '%Y-%m-%dT%H:%M:%S') + timedelta(seconds=timewindow)
        first = 1
    if split[2] == '0x1c88':
        if not split[3] in node_list:
            node_list.append(split[3])
        if not split[4] in node_list:
            node_list.append(split[4])
f.close()
f = open("No_ms\\output4", "r")
fo = open("No_ms\\output4_" + str(timewindow) + "_Seconds_Discrete.csv", "w+")
foc = open("No_ms\\output4_" + str(timewindow) + "_Seconds_Count.csv", "w+")

nodedict = dict()

header = "datetime"
for x in node_list:
    header += "," + x
    nodedict[x] = 0
fo.write(header + "\n")
foc.write(header + "\n")

timestamp = datetime.now()

line = ""
val_string = ""
lensum = 0

for x in f:
    line = f.readline()
    split = line.split()
    timestamp = datetime.strptime(split[0], '%Y-%m-%dT%H:%M:%S')
    if timestamp > delta:
        if lensum > 0:
            val_string = ""
            for v in nodedict.values():
                val_string += str(v) + ','
            val_string = val_string[0:len(val_string) - 1]
            line = timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(lensum) + "," + val_string
            foc.write(line + "\n")
            val_string = ""
            for v in nodedict.values():
                val_string += str(1 if v > 0 else 0) + ','
            val_string = val_string[0:len(val_string)-1]
            line = timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(lensum) + "," + val_string
            fo.write(line + "\n")
            for k in nodedict.keys():
                nodedict[k] = 0
            delta = delta + timedelta(seconds=timewindow)
            line = ""
            lensum = 0
        else:
            val_string = ""
            for v in nodedict.values():
                val_string += str(0) + ','
            val_string = val_string[0:len(val_string)-1]
            line = delta.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(lensum) + "," + val_string
            delta = delta + timedelta(seconds=timewindow)
            foc.write(line + "\n")
            fo.write(line + "\n")
            line = ""
    else:
        if split[2] == '0x1c88':
            if split[3] in nodedict:
                nodedict[split[3]] += 1
                nodedict[split[4]] += 1
                lensum += int(split[1])
val_string = ""
for v in nodedict.values():
    val_string += str(v) + ','
val_string = val_string[0:len(val_string) - 1]
line = delta.strftime("%Y-%m-%dT%H:%M:%S") + "," + val_string
foc.write(line + "\n")
val_string = ''
for v in nodedict.values():
    val_string += str(1 if v > 0 else 0) + ','
val_string = val_string[0:len(val_string)-1]
line = delta.strftime("%Y-%m-%dT%H:%M:%S") + "," + val_string
fo.write(line + "\n")
f.close()
fo.close()
