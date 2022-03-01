import socket
import time
import twisted

UDP_IP = "127.0.0.1"
UDP_PORT = 1534

direct = "C:\\Users\\nickb\\source\\repos\\MastersAI\\ZigbeeFileParser\\FileParsing\\Raw"
file = "data_out_zigbeeout"
f = open(direct + "\\" + file, "r")

for x in f:
    if x.find('[+]') > 0:
        time.sleep(5000)
    sock = socket.socket(socket.AF_INET,  # Internet
    socket.SOCK_DGRAM)  # UDP
    sock.sendto(bytes(x, "utf-8"), (UDP_IP, UDP_PORT))