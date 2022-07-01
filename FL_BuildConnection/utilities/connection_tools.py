def parseID(temp_file):
    for line in open(temp_file, 'r', encoding='utf-8'):
        #print(line)
        if ("Duet Server ID:" in line):
            #print(line[line.index(":") + 2:])
            Duet_ID = line[line.index(":") + 2:]
            break
    return Duet_ID

def detect(temp_file): # return true is f is empty
    return False
