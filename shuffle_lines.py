import random

# Step 1: Read both files into separate lists
with open("predicted_sql.txt", "r") as file1, open("dev_gold.sql", "r") as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()

# Step 2: Generate a list of indices and shuffle it
indices = list(range(len(lines1)))
random.shuffle(indices)

# Step 3: Reorder both lists of lines using the shuffled indices
shuffled_lines1 = [lines1[i] for i in indices]
shuffled_lines2 = [lines2[i] for i in indices]

# Step 4: Write the shuffled lines back to files (or to new files)
with open("shuffled_predicted_sql", "w") as file1, open(
    "shuffled_dev_gold.sql", "w"
) as file2:
    file1.writelines(shuffled_lines1)
    file2.writelines(shuffled_lines2)
