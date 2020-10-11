from datetime import datetime
import csv

file = "3_26_2020"
direct = "C:/Users/nickb/OneDrive/Documents/Masters Project AI/data/"
directory = "Raw"
header = ''
output = ""
first = 0
process = -1
error = -1
pad = 0
Dict = {}
dictIndex = 0

f = open(direct + "\\" + file, "r")

for x in f:
    result = x.find('[+]')
    if first <= 0 and result >= 0:
        delta = datetime.strptime(x[x.find(': ')+2:len(x)-1], '%Y-%m-%d %H:%M:%S')
        delta = delta.timestamp()
        packetTimestamp = datetime.strptime(x[x.find(': ') + 2:len(x) - 1], '%Y-%m-%d %H:%M:%S')
        packetTimestamp = packetTimestamp.timestamp()
        first = 1
    result = x.find('[DstAddr] :')
    if result >= 0:
        addr = x[x.find(': ') + 2:len(x)-2]
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
    result = x.find('<[SrcAddr] :')
    if result >= 0:
        addr = x[x.find(': ') + 2:len(x)-2]
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
        if addr not in Dict.keys() and addr != '0x' and addr != '0xffff' and len(addr) == 6:
            Dict[addr] = dictIndex
            dictIndex += 1
f.close()
deltaTime = [delta] * dictIndex
diffTime = [0] * dictIndex
first = 0

f = open(direct + "\\" + file, "r")
fo = open(directory + "\\" + file + "out", "w+")

wr = csv.writer(fo)
for x in f:
    result = x.find('[+]')
    if result >= 0:
        packetTimestamp = datetime.strptime(x[x.find(': ') + 2:len(x) - 1], '%Y-%m-%d %H:%M:%S')
        packetTimestamp = packetTimestamp.timestamp()
    if x.find("<[DstAddr] :") >= 0:
        addr = x[x.find(': ') + 2:len(x)-2]
        if addr != '0x' and addr != '0xffff':
            dstaddr = x[x.find(': ') + 2:len(x)-2]
        else:
            dstaddr = ''
    elif x.find("<[SrcAddr] :") >= 0:
        addr = x[x.find(': ') + 2:len(x)-2]
        if addr != '0x' and addr != '0xffff':
            srcaddr = x[x.find(': ') + 2:len(x)-2]
        else:
            srcaddr = ''
    elif x.find("IEEE 802.15.4 MAC: -decoding error-") >= 0:
        error = -1
    elif x.find("<[RawData] :") >= 0:
        if error >= 0:
            for y in Dict.keys():
                if y == dstaddr:
                    diffTime[Dict[y]] = 0
                    deltaTime[Dict[y]] = packetTimestamp
                elif y == srcaddr:
                    diffTime[Dict[y]] = 0
                    deltaTime[Dict[y]] = packetTimestamp
                else:
                    diffTime[Dict[y]] = packetTimestamp - deltaTime[Dict[y]]
            wr.writerows([diffTime])
        else:
            error = 0
fo.writelines(output[0:len(output)-1])
f.close()
fo.close()