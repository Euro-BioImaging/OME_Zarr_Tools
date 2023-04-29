git clone -b dev https://git.embl.de/oezdemir/OME_Zarr_Tools.git &&
source OME_Zarr_Tools/install.sh T0XMlxMdq8C6rSxurrdqMqHNrhyhC4f0 dRFXoR852egFtp3lC9NJPYjpPaCBNRa8

### Check what we have at our s3 bucket
mc ls s3minio/ome-zarr-course/
mc ls s3minio/ome-zarr-course/data/

### Copy data to home folder
cd ~;
mc mirror s3minio/ome-zarr-course/data/ ~/data;

### Convert TIFF data locally
batchconvert omezarr --drop_series ~/data/TIFF ~/data/ZARR;

### Convert JPEGs by merging them to a single time series
batchconvert omezarr --drop_series --merge_files --concatenation_order t ~/data/JPEG ~/data/ZARR;

### Convert OIRs by merging them to several time series
batchconvert omezarr --drop_series --merge_files --concatenation_order t ~/data/OIR ~/data/ZARR;


### Visualise locally with napari
napari --plugin napari-ome-zarr ~/data/ZARR/xyzct_8bit__mitosis.ome.zarr

### Visualise locally with fiji
fiji ;



### Convert data at the s3 end
batchconvert omezarr -dt s3 --drop_series data/TIFF data/$USER;
### Convert by merging
batchconvert omezarr -dt s3 --drop_series --merge_files --concatenation_order t data/SERIES_TIFF data/$USER;


### Check whether we have the converted data at the s3 end:
mc ls s3minio/ome-zarr-course/data/ ;

### view via different options
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/xyzct_8bit__mitosis.ome.zarr;
fiji ;
https://kitware.github.io/itk-vtk-viewer/app/?fileToLoad=https://s3.embl.de/ome-zarr-course/data/ZARR/xyz_8bit__nucleus.ome.zarr
# add vizarr and neuroglancer



### Segment data locally 
zseg threshold -m otsu -c 1 -ch 1 -n otsu-c1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr; #TODO fix sampling issue
zseg threshold -m ridler_median -c 1 -ch 1 -n rwm-c1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr; 
zseg threshold -m isodata -c 1 -ch 1 -n iso-c1-ch1 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr; #TODO fix sampling issue


zseg postprocess -m binary_opening -f 2,2 -l otsu-1-ch0 ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr;

napari --plugin napari-ome-zarr ~/data/ZARR/xyc_8bit__tub_h2b_cecog_battery.ome.zarr

### Segment data remotely 
zseg threshold -m otsu -c 1 -n otsu-1 https://s3.embl.de/ome-zarr-course/data/ZARR/xyz_8bit_calibrated__fib_sem_crop_original.ome.zarr






