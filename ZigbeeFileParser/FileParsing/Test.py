txt = "IEEE 802.15.4 MAC:\n[[[ IEEE 802.15.4 ]]]\n### [MAC] ###\n <[Res] : 0b0>\n <[IntraPAN] : '1 : True'>\n <[AckReq] : '0 : False'>\n <[FramePending] : '0 : False'>\n <[Security] : '0 : False'>\n <[Type] : '1 : Data'>\n <[SrcAddrMode] : '2 : 16-bit address'>\n <[FrameVers] : 0b00>\n <[DstAddrMode] : '2 : 16-bit address'>\n <[Res] : 0b00>\n <[SeqNum] : 202>\n <[DstPANID] : 0x1c88>\n <[DstAddr] : 0xffff>\n <[SrcPANID] : 0x>\n <[SrcAddr] : 0x0000>\n\t### [Data] ###\n\t <[RawData] : 0x0912fcff00000164881c00ffff2e210028e6e55901881c00ffff2e21000048e9bcfd4254825d4f>\n"
import json
from datetime import datetime

data = {}
json_data = json.dumps(data)

# setting the maxsplit parameter to 1, will return a list with 2 elements!
print(txt)
x = txt.split("\n")
for item in x:
    if item.find('<') > 0:
        error = 0
        if x.find('[+]') >= 0:
            data = {}
            packetTimestamp = datetime.strptime(msg[msg.find(': ') + 2:len(msg) - 1], '%Y-%m-%d %H:%M:%S')
            data["TimeStamp"] = packetTimestamp
        if x.find("IEEE 802.15.4 MAC: -decoding error-") >= 0:
            error = -1
        if x.find("IEEE 802.15.4 frame:") >= 0:
            if error >= 0:
                x = x.split("\n")
                for item in x:
                    if item.find('<') > 0:
                        item = item.replace('<', '')
                        item = item.replace('>', '')
                        item = item.replace('\'', '')
                        split = item.split(":")
                        split[0] = split[0].replace('[', '')
                        split[0] = split[0].replace(']', '')
                        split[0] = split[0].strip()
                        split[1] = split[1].strip()
                        data[split[0]] = split[1]
                        json_data = json.dumps(data)
                    sensorReadings = 'sensorReadings'
                    obj = Networkif.objects.first()
                    sensorReadings_value = getattr(obj, sensorReadings + json_data)
                    obj.sensorReadings = sensorReadings_value
                    obj.save()
