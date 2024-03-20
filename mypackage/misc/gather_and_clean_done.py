# this script is ran after the parrallel qats grid, to gather all the temporary done file in each output light curve sub folder, write it in the main done file and delete the temporary done file.
# this is a solution to avoid the parrallel file writing that proves to be problematic

import sys
import os

output_folders=sys.argv[1]

print("clean script print:", output_folders)

done_lc_arguments = []
for folders in os.listdir(output_folders):
    if ".txt" in folders:
        continue
    
    temp_done_file = os.path.join(output_folders, folders, "temp_done.txt")
    
    try:
        with open(temp_done_file, "r") as f:
            
            #contents = f.read()
            done_lc_arguments.append(f.read())
        
        os.remove(temp_done_file)
    except FileNotFoundError:
        continue
        

print("write into main done")
done_file = os.path.join(output_folders, "done.txt")

with open(done_file, "a") as f:
    for c in done_lc_arguments:
        f.write(c)
print("writing done")

