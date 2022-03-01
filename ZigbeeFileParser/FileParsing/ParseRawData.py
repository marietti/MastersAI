from datetime import datetime
import csv

file = "data_out_zigbee"
#direct = "C:/Users/nickb/OneDrive/Documents/Masters Project AI/data/"
direct = "C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Raw"
header = ''
first = 0
process = -1
error = -1
pad = 0
keys = ['TimeStamp', 'Len', 'IntraPAN', 'AckReq', 'FramePending', 'Security', 'Type', 'SrcAddrMode', 'FrameVers', 'DstAddrMode', 'SeqNum', 'DstPANID', 'DstAddr', 'SrcPANID', 'SrcAddr']
output = {}
dictIndex = 0

f = open(direct + "\\" + file, "r")
fo = open(direct + "\\" + file + "out", 'w', newline='', encoding='utf-8')
wr = csv.DictWriter(fo, fieldnames=keys)
wr.writeheader()

for x in keys:
    output[x] = ""

for x in f:
    result = x.find('[+]')
    if result >= 0:
        if first > 0:
            wr.writerow(output)
        else:
            first = 1
        packetTimestamp = datetime.strptime(x[x.find(': ') + 2:len(x) - 1], '%Y-%m-%d %H:%M:%S')
        output["TimeStamp"] = packetTimestamp
        error = 0
    if x.find("IEEE 802.15.4 MAC: -decoding error-") >= 0:
        error = -1
    if x.find("IEEE 802.15.4 frame:") >= 0:
        val = x[x.find(': ') + 2:len(x)-2]
        output["Len"] = len(val)
    if error >= 0:
        for value in keys:
            if x.find("<[" + value + "] :") >= 0:
                if x.find("'") >= 0:
                    index1 = x.find("'")+1
                    index2 = x.find(':', x.find("'"))
                    val = x[x.find("'")+1:index2]
                    output[value] = val
                else:
                    val = x[x.find(': ') + 2:len(x)-2]
                    output[value] = val
f.close()
fo.close()
