import time
from datetime import datetime, date, time, timedelta
file = "cc2531_sniffer_2_3_2021out"
#file = "3_23_2021out"
folder = "Raw"
timewindow = [5, 10, 45, 60, 90, 120, 180, 240, 300]
#11:34 Entered
#11:58 Exited
#12:00 Entered
#12:50 Entered
#12:51 Exited
#12:54 Entered
#1:06 Exited
#1:10? Entered
#4:30? Exited

#occupied_windows = ["2021-03-23 11:34:00", "2021-03-23 11:58:00", "2021-03-23 12:00:00", "2021-03-23 12:50:00", "2021-03-23 12:54:00", "2021-03-23 13:06:00", "2021-03-23 13:10:00", "2021-03-23 16:30:00"]
occupied_windows = ["2021-02-03 10:35:00", "2021-02-03 11:30:00", "2021-02-03 11:56:00", "2021-02-03 12:52:00",
                    "2021-02-03 01:05:00", "2021-02-03 15:09:00", "2021-02-03 15:20:00", "2021-02-03 16:45:00"]

for z in timewindow:
    first = 0
    node_list = []
    node_discrete_list = []
    f = open(folder + "\\" + file, "r")

    for x in f:
        split = x.rstrip().split(',')
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
    f = open(folder + "\\" + file, "r")
    foc = open(folder + "\\" + file + "_" + str(z) + "_Seconds_Count.csv", "w+")

    nodedict = dict()

    header = "datetime,len_sum,smallest,largest,count_sum,occupied"
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
    occupied = 0
    occupied_windows_index = 0
    occupied_windows_datetime = datetime.strptime(occupied_windows[occupied_windows_index], '%Y-%m-%d %H:%M:%S')

    for x in f:
        split = x.rstrip().split(',')
        if not split[0] == "TimeStamp":
            timestamp = datetime.strptime(split[0], '%Y-%m-%d %H:%M:%S')
            if timestamp > occupied_windows_datetime and occupied_windows_index < len(occupied_windows):
                if occupied > 0:
                    occupied = 0
                else:
                    occupied = 1
                occupied_windows_index += 1
                if occupied_windows_index < len(occupied_windows):
                    occupied_windows_datetime = datetime.strptime(occupied_windows[occupied_windows_index],
                                                                  '%Y-%m-%d %H:%M:%S')
            if timestamp > delta:
                if lensum > 0:
                    val_string = ""
                    for v in nodedict.values():
                        val_string += str(v) + ','
                        count_sum += v
                    val_string = val_string[0:len(val_string) - 1]
                    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + str(occupied) + "," + val_string
                    foc.write(line + "\n")
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
                        lensum += int(split[1])
                else:
                    val_string = ""
                    for v in nodedict.values():
                        val_string += str(0) + ','
                    val_string = val_string[0:len(val_string)-1]
                    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(0) + "," + str(0) + "," + str(count_sum) + "," + val_string + "," + str(occupied)
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
                    lensum += int(split[1])

    val_string = ""
    for v in nodedict.values():
        val_string += str(v) + ','
    val_string = val_string[0:len(val_string) - 1]
    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + str(occupied) + "," + val_string
    foc.write(line + "\n")
    val_string = ""
    for v in nodedict.values():
        val_string += str(1 if v > 0 else 0) + ','
    val_string = val_string[0:len(val_string) - 1]
    line = delta.strftime("%Y-%m-%d %H:%M:%S") + "," + str(lensum) + "," + str(smallest) + "," + str(largest) + "," + str(count_sum) + "," + str(occupied) + "," + val_string
    f.close()
    foc.close()
