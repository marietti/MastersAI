f = open("No_ms\\output4", "r")
fo = open("No_ms\\output4.csv", "w+")
filecount = 0
fo.write("datetime,len")
for x in f:
    line = f.readline()
    split = line.split()
    fo.write(split[0] + "," + split[1] + "\n")
f.close()
fo.close()
