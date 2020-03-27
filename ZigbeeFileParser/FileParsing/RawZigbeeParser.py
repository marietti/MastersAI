file = "3_26_2020"
directory = "Raw"
header = "datetime,channel,RSSI,frame,len,res,intrapan,ackreq,framepending,security,type,srcaddrmode,framevers," \
         "dstaddrmode,res,seqnum,dstpanid,dstaddr,srcpanid,srcaddr,rawdata\n"
output = ""
process = -1
pad = 0
f = open(directory + "\\" + file, "r")
fo = open(directory + "\\" + file + "out", "w+")
fo.writelines(header)
for x in f:
    print(x)
    result = x.find('[+]')
    if x.find("RawData") >= 0:
        pad = -1
    if len(x.strip()) == 0:
        if pad == 0:
            output += "0x0,"
        fo.writelines(output[0:len(output)-1]+'\n')
        output = ""
        process = -1
        pad = 0
    if x.find("IEEE 802.15.4 frame: ") >= 0:
        output += x[x.find(": ")+2:len(x)-1] + ","
        output += str(len(x[x.find(": ")+2:len(x)-1])*4) + ","
    elif x.find("channel: ") >= 0:
        output += x[x.find(": ") + 2:x.find(",")] + ","
    elif x.find("RSSI: ") >= 0:
        output += x[x.find(": ") + 2:x.find(",")] + ","
    elif x.find("### [MAC] ###") >= 0:
        process = 1
    elif result >= 0:
        print(x[x.find(': ')+2:len(x)])
        output += x[x.find(': ')+2:len(x)-1] + ","
    elif process >= 0 and x.find("### [Data] ###") == -1:
        val = x[x.find('<')+1:x.find('>')]
        idVal = val[val.find('[')+1:val.find(']')]
        if val[val.find(': ') + 2:len(val)].find("'") >= 0:
            sub = val[val.find(': ') + 2:len(val)]
            if sub[1:sub.find(":")-1] != "0x":
                output += sub[1:sub.find(":")-1] + ","
            else:
                output += "0x0,"
        else:
            if val[val.find(': ') + 2:len(val)] != "0x":
                output += val[val.find(': ') + 2:len(val)] + ","
            else:
                output += "0x0,"

f.close()
fo.close()
