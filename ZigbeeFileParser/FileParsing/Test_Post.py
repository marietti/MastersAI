import requests
import json
import time
from datetime import datetime

UDP_IP = "127.0.0.1"
UDP_PORT = 1534

direct = "C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Raw"
file = "data_out_zigbee"
f = open(direct + "\\" + file, "r")
SOCK_ADDR_REM = '137.190.19.198'

data = {}
lastTimeStamp = datetime
packetTimestamp = datetime
first = True
for line in f:
    data = {}
    if line.find('[+]') > 0:
        packetTimestamp = datetime.strptime(line[line.find(': ') + 2:len(line) - 2], '%Y-%m-%d %H:%M:%S')
        data["TimeStamp"] = str(packetTimestamp)
        if first:
            lastTimeStamp = packetTimestamp
            true = False
        json_data = json.dumps(data)
        out_data = json.loads(json_data)
        r = requests.post('http://' + UDP_IP + ':8000/interface/', data=out_data)
        r_dictionary = r.json()
        print(r_dictionary)
    elif line.find('IEEE 802.15.4 MAC') > 0:
        line = line.split("\\n")
        error = 0
        for item in line:
            if item.find("IEEE 802.15.4 MAC: -decoding error-") >= 0:
                error = -1
                send = 0
            if item.find('<') > 0:
                send = 1
                if error >= 0:
                    msg = item.split("\n")
                    for msg_item in msg:
                        if item.find('<') > 0:
                            msg_item = msg_item.replace('<', '')
                            msg_item = msg_item.replace('>', '')
                            msg_item = msg_item.replace('\'', '')
                            msg_item = msg_item.replace('\\t', '')
                            split = msg_item.split(":")
                            if len(split) > 1:
                                split[0] = split[0].replace('[', '')
                                split[0] = split[0].replace(']', '')
                                split[0] = split[0].strip()
                                split[1] = split[1].strip()
                                data[split[0]] = split[1]
        json_data = json.dumps(data)
        out_data = json.loads(json_data)
        print(out_data)
        difference = (packetTimestamp - lastTimeStamp)
        time.sleep(difference.total_seconds())
        lastTimeStamp = packetTimestamp
        r = requests.post('http://' + UDP_IP + ':8000/interface/', data=out_data)
        r_dictionary = r.json()
        print(r_dictionary)
