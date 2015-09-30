PRINT, 'convert ascii file to hdf5'
testfile = '/Users/users/breddels/vaex/src/SubspaceFinding/data/helmi2000.asc'
h5file_id = H5F_CREATE('/tmp/test.hdf5')

N = 3300000; nr of rows

h5group_columns = H5G_CREATE(h5file_id, "columns") ; for vaex, all columns should be grouped under columns
h5type_id = H5T_IDL_CREATE(1.0d) ; create double datatype
h5data_id = H5S_CREATE_SIMPLE(N)

h5_E = H5D_CREATE(h5group_columns, 'E', h5type_id, h5data_id)
h5_L = H5D_CREATE(h5group_columns, 'L', h5type_id, h5data_id)
h5_Lz = H5D_CREATE(h5group_columns, 'Lz', h5type_id, h5data_id)

dataspace = H5D_GET_SPACE(h5_E)



FREE_LUN, 1
OPENR, 1, testfile

index = 0L
WHILE NOT EOF(1) DO BEGIN
  READF, 1, E,L,Lz
  if (index MOD 100000) EQ 0 then  begin
    print, index, ' of',N 
  end
  H5S_SELECT_HYPERSLAB, dataspace, [index], [1], stride=[1], /RESET
  memory_space_id = H5S_CREATE_SIMPLE([1])
  H5D_WRITE, h5_E, [E], MEMORY_SPACE_ID=memory_space_id,  FILE_SPACE_ID=dataspace
  H5D_WRITE, h5_L, [L], MEMORY_SPACE_ID=memory_space_id,  FILE_SPACE_ID=dataspace
  H5D_WRITE, h5_Lz, [Lz], MEMORY_SPACE_ID=memory_space_id,  FILE_SPACE_ID=dataspace
  index = index + 1
ENDWHILE

H5F_CLOSE, h5file_id
FREE_LUN, 1
  
end