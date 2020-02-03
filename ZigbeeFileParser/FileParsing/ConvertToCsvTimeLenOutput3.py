f = open("No_ms\\output3", "r")
fo = open("No_ms\\output3.csv", "w+")
filecount = 0
fo.write("datetime,len")
for x in f:
    line = f.readline()
    split = line.split()
    fo.write(split[0].endswith('.%f') + "," + split[1] + "\n")
f.close()
fo.close()
