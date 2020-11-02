import os
from shutil import copyfile
load_path = r"C:\Users\randy\Desktop\暫存"
save_path = r"C:\Users\randy\Downloads\the_way_to_train"
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(load_path):
    for file in f:
        if '.pdf' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    base = os.path.basename(f)
    print(base)
    copyfile(f, "%s/%s"%(save_path,base))