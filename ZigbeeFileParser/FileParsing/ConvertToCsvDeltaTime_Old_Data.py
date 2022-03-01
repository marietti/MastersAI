from datetime import datetime
import csv

file = "Calendar_11_15_9_1"
#direct = "C:/Users/nickb/OneDrive/Documents/Masters Project AI/data/"
direct = "C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\No_ms"
directory = "No_ms"
header = ''
output = ""
first = 0
process = -1
error = -1
pad = 0
Dict = {'time_stamp': 0}
dictIndex = 1

f = open(direct + "\\" + file, "r")

for x in f:
    split = x.split()
    if first <= 0:
        delta = datetime.strptime(split[0], '%Y-%m-%dT%H:%M:%S')
        delta = delta.timestamp()
        packetTimestamp = datetime.strptime(split[0], '%Y-%m-%dT%H:%M:%S')
        packetTimestamp = packetTimestamp.timestamp()
        first = 1
    if split[2] == '0x1c88':
        addr = split[3]
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
        addr2 = split[4]
        if addr not in Dict.keys() and addr2 != '0x' and addr2 != '0xffff' and len(addr2) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
        if addr not in Dict.keys() and addr2 != '0x' and addr2 != '0xffff' and len(addr2) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1

f.close()
deltaTime = [delta] * dictIndex
diffTime = [0] * dictIndex
first = 0

f = open(direct + "\\" + file, "r")
fo = open(direct + "\\" + file + "out", 'w', newline='', encoding='utf-8')
for x in Dict.keys():
    header += "," + x
fo.write(header + "\n")
wr = csv.writer(fo)
for x in f:
    split = x.split()
    packetTimestamp = datetime.strptime(split[0], '%Y-%m-%dT%H:%M:%S')
    diffTime[0] = packetTimestamp
    packetTimestamp = packetTimestamp.timestamp()
    if split[2] == '0x1c88':
        for y in Dict.keys():
            index = Dict[y]
            if index != 0:
                if y == split[3] and split[3] != '0x' and split[3] != '0xffff' and len(split[3]) == 6:
                    diffTime[index] = 0
                    deltaTime[index] = packetTimestamp
                elif y == split[4] and split[4] != '0x' and split[4] != '0xffff' and len(split[4]) == 6:
                    diffTime[index] = 0
                    deltaTime[index] = packetTimestamp
                elif split[3] != '0x' and split[3] != '0xffff' and len(split[3]) == 6 and split[4] != '0x' and split[4] != '0xffff' and len(split[4]) == 6:
                    diffTime[index] = packetTimestamp - deltaTime[index]
    wr.writerows([diffTime])
fo.writelines(output[0:len(output)-1])
f.close()
fo.close()