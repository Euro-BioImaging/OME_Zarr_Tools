git clone -b dev https://git.embl.de/oezdemir/OME_Zarr_Tools.git &&
source OME_Zarr_Tools/install.sh T0XMlxMdq8C6rSxurrdqMqHNrhyhC4f0 dRFXoR852egFtp3lC9NJPYjpPaCBNRa8

### Check what we have at our s3 bucket
mc tree s3minio/ome-zarr-course/
mc ls s3minio/ome-zarr-course/data/MFF/
mc ls s3minio/ome-zarr-course/data/JPEG/

### Convert the MFF data at the s3 end
batchconvert omezarr -st s3 -dt s3 --drop_series data/MFF data/ZARR/$USER;

### Convert the JPEGs by merging them to a single time series
batchconvert omezarr -st s3 -dt s3 --drop_series --merge_files --concatenation_order t data/JPEG data/ZARR/$USER;

### Check what has changed at the s3 end:
mc tree -d 2 s3minio/ome-zarr-course/

### Copy the Zarr data to the home folder
mc mirror s3minio/ome-zarr-course/data/ZARR/$USER ~/data/ZARR;

### Visualise locally and remotely with napari
napari --plugin napari-ome-zarr ~/data/ZARR/xyzct_8bit__mitosis.ome.zarr
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/xyzct_8bit__mitosis.ome.zarr

### Visualise locally with fiji
fiji ;

### Other viewing options
https://kitware.github.io/itk-vtk-viewer/app/?fileToLoad=https://s3.embl.de/ome-zarr-course/data/ZARR/xyz_8bit__nucleus.ome.zarr
# add vizarr and neuroglancer



### Segment data locally 
zseg threshold -m otsu -c 1 -ch 1 -n otsu-c1-ch1 ~/data/ZARR/23052022_D3_0002_positiveCTRL.ome.zarr;

zseg threshold -m ridler_median -c 1 -ch 1 -n rwm-c1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr; 
zseg threshold -m li -c 1 -ch 1 -n iso-c1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr; #TODO fix sampling issue

zseg postprocess -m binary_opening -f 2,2 -l rwm-1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr;

napari --plugin napari-ome-zarr ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr

### Segment data remotely 
zseg threshold -m otsu -c 1 -n otsu-1 https://s3.embl.de/ome-zarr-course/data/ZARR/xyz_8bit_calibrated__fib_sem_crop_original.ome.zarr


batchconvert omezarr --drop_series --merge_files --concatenation_order t ~/data/JPEG ~/data/ZARR;


