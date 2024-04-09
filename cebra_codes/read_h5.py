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
