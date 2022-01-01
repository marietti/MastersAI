import time
from datetime import datetime, date, time, timedelta
file = "cc2531_sniffer_2_3_2021out"
timewindow = [45]

for z in timewindow:
    first = 0
    node_list = []
    node_discrete_list = []
    f = open("Raw\\" + file, "r")

    for x in f:
        split = x.split(',')
        if not split[0] == "TimeStamp":
            if first <= 0:
                delta = datetime.strptime(split[0], '%Y-%m-%d %H:%M:%S') + timedelta(seconds=z)
                first = 1
            if split[11] == '0x1c88':
                if not split[12] in node_list:
                    node_list.append(split[12])
                if not split[14] in node_list:
                    node_list.append(split[14])
    f.close()
    f = open("Raw\\" + file, "r")
    foc = open("Raw\\" + file + "_" + str(z) + "_Seconds_Count.csv", "w+")

    nodedict = dict()

    header = "datetime,len_sum,smallest,largest,count_sum"
    for x in node_list:
        header += "," + x.rstrip()
        nodedict[x] = 0
    foc.write(header + "\n")

    timestamp = datetime.now()

    line = ""
    val_string = ""
    lensum = 0
    smallest = 100000
    largest = 0
    count_sum = 0

    for x in f:
        split = x.split(',')
        if not split[0] == "TimeStamp":
            timestamp = datetime.strptime(split[0], '%Y-%m-%d %H:%M:%S')
            if timestamp > delta:
                if lensum > 0:
                    val_string = ""
                    for v in nodedict.values():
                        val_string += str(v) + ','
                    val_string = val_string[0:len(val_string) - 1]
                    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + val_string
                    foc.write(line + "\n")
                    val_string = ""
                    for v in nodedict.values():
                        val_string += str(1 if v > 0 else 0) + ','
                    val_string = val_string[0:len(val_string)-1]
                    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + val_string
                    for k in nodedict.keys():
                        nodedict[k] = 0
                    delta = delta + timedelta(seconds=z)
                    line = ""
                    lensum = 0
                    smallest = 100000
                    largest = 0
                    count_sum = 0
                    if split[11] == '0x1c88':
                        if int(split[1]) < smallest:
                            smallest = int(split[1])
                        if int(split[1]) > largest:
                            largest = int(split[1])
                        nodedict[split[12]] += 1
                        nodedict[split[14]] += 1
                        count_sum += 1
                        lensum += int(split[1])
                else:
                    val_string = ""
                    for v in nodedict.values():
                        val_string += str(0) + ','
                    val_string = val_string[0:len(val_string)-1]
                    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(0) + "," + str(0) + "," + str(count_sum) + "," + val_string
                    delta = delta + timedelta(seconds=z)
                    foc.write(line + "\n")
                    line = ""
            else:
                if split[11] == '0x1c88':
                    if int(split[1]) < smallest:
                        smallest = int(split[1])
                    if int(split[1]) > largest:
                        largest = int(split[1])
                    nodedict[split[12]] += 1
                    nodedict[split[14]] += 1
                    count_sum += 1
                    lensum += int(split[1])

    val_string = ""
    for v in nodedict.values():
        val_string += str(v) + ','
    val_string = val_string[0:len(val_string) - 1]
    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + val_string
    foc.write(line + "\n")
    val_string = ""
    for v in nodedict.values():
        val_string += str(1 if v > 0 else 0) + ','
    val_string = val_string[0:len(val_string) - 1]
    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + val_string
    f.close()
    foc.close()
