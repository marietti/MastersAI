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
keys = ['TimeStamp', 'Len', 'IntraPAN', 'AckReq', 'FramePending', 'Security', 'Type', 'SrcAddrMode', 'FrameVers', 'DstAddrMode', 'SeqNum', 'DstPANID', 'DstAddr', 'SrcPANID', 'SrcAddr', 'RawData']
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
        packetTimestamp = datetime.strptime(x[x.find(': ') + 2:len(x) - 2], '%Y-%m-%d %H:%M:%S')
        output["TimeStamp"] = packetTimestamp
        error = 0
    if x.find("IEEE 802.15.4 MAC: -decoding error-") >= 0:
        error = -1
    if x.find("IEEE 802.15.4 MAC:") >= 0:
        if error >= 0:
            x = x.replace('\\t', '')
            split_new = x.split('\\n')
            for item in split_new:
                if item.find('<') > 0:
                    item = item.replace('<', '')
                    item = item.replace('>', '')
                    item = item.replace('\'', '')
                    split = item.split(":")
                    split[0] = split[0].replace('[', '')
                    split[0] = split[0].replace(']', '')
                    split[0] = split[0].strip()
                    split[1] = split[1].strip()
                    if split[0] == 'RawData':
                        output["Len"] = len(split[1]) - 2
                    if split[0] in output.keys():
                        output[split[0]] = split[1]
f.close()
fo.close()
