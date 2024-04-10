import h5py


f_name='manif_file_0.hdf5'

# Just print names in the command #
def print_names(name, obj):
    print(name)
    if isinstance(obj, h5py.Group):
        print(f"{name} is a group")
    elif isinstance(obj, h5py.Dataset):
        print(f"{name} is a dataset")
    else:
        print(f"{name} Unknown type")

# 
with h5py.File(f_name, 'r') as file:
    file.visititems(print_names)
     

### Save database structure in a txt

#output_file = 'list_groups_and_db_output.txt'

#with h5py.File(f_name, 'r') as file:
  #  with open(output_file, 'w') as out:
 #       def print_names(name, obj):
 #           line = f"{name}\n"
 #           out.write(line)
 #           if isinstance(obj, h5py.Group):
 #               out.write(f"{name} is a group\n")
  #          elif isinstance(obj, h5py.Dataset):
 #               out.write(f"{name} is a db_output\n")
 #           else:
  #              out.write(f"{name} Unkn. type\n")

 #       file.visititems(print_names)

#print(f"Names are saved in  {output_file}.")

with h5py.File(f_name, 'r') as file:
    # Access a specific dataset
    group_= file['manif_mod_div_et_imp']
    print(f"Content of {group_.name}:")
    for name in group_:
        print(name)

    # Aceess  a  dataset
    #dataset = file['path/to/dataset']

    # Read a Dataset
    #data = dataset[:]
    # Get dataset shape
    #shape = dataset.shape
    
    # Get dataset data type
    # dtype = dataset.dtype
    
    # Print information
    #print(f"Shape: {shape}, Type: {dtype}")

    # Now, data contains the dataset's contents, and you can process it as needed
    print(data)