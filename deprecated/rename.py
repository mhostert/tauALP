import os
import re

# Directory where the files are stored
directory = "pythia8_cluster"

for case in ["soft", "hard", "both"]:
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Match pattern like: tau_events_NuMI_120GeV_20_hard.txt
        match = re.match(rf"(tau_events_.*?)(_)(\d+)_{case}\.txt", filename)
        if match:
            base, _, number = match.groups()
            new_filename = f"{base}_{case}_{number}.txt"

            # Full pathscd
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            print(f"Renaming: {filename} -> {new_filename}")
            os.rename(old_path, new_path)
