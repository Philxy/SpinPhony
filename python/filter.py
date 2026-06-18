import csv
import math

def filter_neighbors_by_distance(input_file, output_file, cutoff):
    """
    Reads a file containing atomic coordinates and exchange couplings,
    and writes out only the lines where the distance between atoms i-j 
    and i-k are BOTH less than or equal to the cutoff distance.
    """
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        # Using csv module to cleanly handle comma-separated values
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 1. Read and immediately write the header to the new file
        try:
            header = next(reader)
            writer.writerow(header)
        except StopIteration:
            print("The input file is empty.")
            return

        valid_lines_count = 0
        skipped_lines_count = 0

        # 2. Iterate through the remaining data lines
        for row in reader:
            try:
                # Extract the first 6 columns as floats
                x_ij, y_ij, z_ij = float(row[0]), float(row[1]), float(row[2])
                x_ik, y_ik, z_ik = float(row[3]), float(row[4]), float(row[5])
                
                # Calculate the magnitude (distance) of both relative vectors
                r_ij = math.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
                r_ik = math.sqrt(x_ik**2 + y_ik**2 + z_ik**2)
                
                # 3. Apply the cutoff condition
                if r_ij <= cutoff and r_ik <= cutoff:
                    writer.writerow(row)
                    valid_lines_count += 1
                else:
                    skipped_lines_count += 1
                    
            except (ValueError, IndexError):
                # Failsafe for empty lines or malformed rows
                print(f"Skipping malformed row: {row}")
                
        print(f"Filtering complete!")
        print(f"Lines kept: {valid_lines_count}")
        print(f"Lines dropped (exceeded cutoff {cutoff}): {skipped_lines_count}")

# ==========================================
# Script Execution
# ==========================================
if __name__ == "__main__":
    # Define your parameters here
    INPUT_FILENAME = 'Inputs/CrI3/CrI3/transformed_SLC_tensor_z.csv'   # Replace with your actual input file name
    OUTPUT_FILENAME = 'Inputs/CrI3/transformed_SLC_tensor_z_filtered.csv'  # Replace with your desired output file name
    CUTOFF_DISTANCE = 1.0                  # Cutoff distance in units of the lattice constant
    
    filter_neighbors_by_distance(INPUT_FILENAME, OUTPUT_FILENAME, CUTOFF_DISTANCE)