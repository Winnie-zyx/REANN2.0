# Define the input file name
input_file = 'configuration'

# Read the file and process it
with open(input_file, 'r') as file:
    lines = file.readlines()

# Iterate through the lines and find the one with "abprop"
for i, line in enumerate(lines):
    if line.startswith('abprop'):
        # Split the line into a list of values
        values = line.split()

        # Multiply the last nine values by -1
        for j in range(1, 10):
            values[-j] = str(float(values[-j]) * -1)

        # Replace the original line with the updated one
        lines[i] = ' '.join(values) + '\n'
        #break  # We only need to modify the first line with 'abprop'

# Write the modified content back to the file
with open(input_file, 'w') as file:
    file.writelines(lines)

print(f"File '{input_file}' updated successfully.")
