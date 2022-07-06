import re

def parseID(temp_file):
    Duet_ID = ""
    for line in open(temp_file, 'r+', encoding='utf-8'):
        #print(line)
        if ("Duet Client ID:" in line or "Duet Server ID:" in line):
            #print(line[line.index(":") + 2:])
            Duet_ID = line[line.index(":") + 2:]
            break
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    Duet_ID_result = ansi_escape.sub('', Duet_ID)
    Duet_ID_result = Duet_ID_result.strip('\n')
    return Duet_ID_result

