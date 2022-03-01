import time
from datetime import datetime, date, time, timedelta
file = "cc2531_sniffer_2_3_2021"
timewindow = 5
first = 0
node_list = []
node_discrete_list = []
f = open("Raw\\" + file, "r")

for x in f:
    split = x.split(',')
    if first <= 0:
        delta = datetime.strptime(split[0], '%Y-%m-%d %H:%M:%S') + timedelta(seconds=timewindow)
        first = 1
    if not split[17] in node_list:
        node_list.append(split[17])
    if not split[19] in node_list:
        node_list.append(split[19])
f.close()
f = open("Raw\\" + file, "r")
fo = open("Raw\\" + file + "_" + str(timewindow) + "_Seconds_Discrete.csv", "w+")
foc = open("Raw\\" + file + "_" + str(timewindow) + "_Seconds_Count.csv", "w+")

nodedict = dict()

header = "datetime,len_sum"
for x in node_list:
    header += "," + x
    nodedict[x] = 0
fo.write(header + "\n")
foc.write(header + "\n")

timestamp = datetime.now()

line = ""
val_string = ""
lensum = 1

for x in f:
    split = x.split(',')
    timestamp = datetime.strptime(split[0], '%Y-%m-%d %H:%M:%S')
    if timestamp > delta:
        if lensum > 1:
            val_string = ""
            for v in nodedict.values():
                val_string += str(v) + ','
            val_string = val_string[0:len(val_string) - 1]
            line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + val_string
            foc.write(line + "\n")
            val_string = ""
            for v in nodedict.values():
                val_string += str(2 if v > 1 else 1) + ','
            val_string = val_string[0:len(val_string)-1]
            line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + val_string
            fo.write(line + "\n")
            for k in nodedict.keys():
                nodedict[k] = 0
            delta = delta + timedelta(seconds=timewindow)
            line = ""
            lensum = 1
            nodedict[split[17]] += 1
            nodedict[split[19]] += 1
            lensum += int(split[1])
        else:
            val_string = ""
            for v in nodedict.values():
                val_string += str(0) + ','
            val_string = val_string[0:len(val_string)-1]
            line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + val_string
            delta = delta + timedelta(seconds=timewindow)
            foc.write(line + "\n")
            fo.write(line + "\n")
            line = ""
    else:
        nodedict[split[17]] += 1
        nodedict[split[19]] += 1
        lensum += int(split[1])

val_string = ""
for v in nodedict.values():
    val_string += str(v) + ','
val_string = val_string[0:len(val_string) - 1]
line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + val_string
foc.write(line + "\n")
val_string = ""
for v in nodedict.values():
    val_string += str(2 if v > 1 else 1) + ','
val_string = val_string[0:len(val_string) - 1]
line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + val_string
fo.write(line + "\n")
f.close()
fo.close()
foc.close()
