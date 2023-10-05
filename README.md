# OME_Zarr_Tools

## About

This repository provides material and guidance for working with image data stored in OME-Zarr format (and optionally in S3 buckets). 

## Software installation

Please clone this repository and build the environment using the following command:

`mamba env create -f OME-Zarr-Tools/envs/environment.yml`

Activate the environment:

`mamba activate ome_zarr_env`

## Practical

Most of the demonstrated tools are mainly command line applications. So the commands given in the sections below can be copy-pasted to a terminal window and executed.

### Inspection of the remote datasets

Check out what we have at our s3 bucket:

``` 
mc tree -d 3 s3/ome-zarr-course/
``` 
``` 
mc ls s3/ome-zarr-course/data/MFF/
``` 
``` 
mc ls s3/ome-zarr-course/data/JPEG/
```
``` 
mc ls s3/ome-zarr-course/data/ZARR/common/
```

Check out the multiscales metadata for one of the existing OME-Zarr datasets:
``` 
mc cat s3/ome-zarr-course/data/ZARR/common/13457537T.zarr/.zattrs
```

Check out the array metadata for the highest resolution array:
``` 
mc cat s3/ome-zarr-course/data/ZARR/common/13457537T.zarr/0/.zarray
```

```
ome_zarr info https://s3.embl.de/ome-zarr-course/data/ZARR/common/13457537T.zarr
```

### Creation of OME-Zarr from remote data

The remote datasets can be converted in a parallelised manner by using the `batchconvert` tool. 

#### First check out what data we have the s3 end:
```
mc tree -d 2 s3/ome-zarr-course/
```

#### Independent conversion of the input files:
The followin command will map each input file in the `data/MFF` folder to a single OME-Zarr series, which will be located in a specific directory for each user. 

```
batchconvert omezarr -st s3 -dt s3 --drop_series data/MFF data/ZARR/$USER;
```
Note that the `-st s3` option will make sure that the input path is searched for in the s3 bucket, while `-dt s3` will trigger the output files to be transferred to the s3 bucket under the output path.

#### Grouped conversion mode:

Another conversion mode will assume that the input files are part of the same series and thus will merge them along a specific axis during the conversion process.
```
batchconvert omezarr -st s3 -dt s3 --drop_series --merge_files --concatenation_order t data/JPEG data/ZARR/$USER;
```
The `merge_files` flag will ensure the grouped conversion option and the `--concatenation_order t` option will make sure that the files will be merged along the time channel. 

#### Check what has changed at the s3 end after the conversion:
```
mc tree -d 2 s3/ome-zarr-course/
```
```
mc ls s3/ome-zarr-course/data/ZARR/$USER/
```

#### Copy the converted Zarr data to the home folder
```
mc mirror s3/ome-zarr-course/data/ZARR/$USER ~/data/ZARR;
```

### Visualisation

#### Napari

Visualise the remote data using Napari together with the napari-ome-zarr plugin.
```
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/xyzct_8bit__mitosis.ome.zarr
```
Optional: visualise the local OME-Zarr data:
```
napari --plugin napari-ome-zarr ~/data/ZARR/xyzct_8bit__mitosis.ome.zarr
```
Optional: visualise big remote OME-Zarr data:
```
napari --plugin napari-ome-zarr https://s3.embl.de/i2k-2020/platy-raw.ome.zarr
```


#### Fiji
fiji ;
[ Plugins > BigDataViewer > OME-Zarr > Open OME-Zarr from S3...]

Visualise the self-created OME-Zarr: 
Note that you need to first replace $USER with your user name in the below url.
S3 URL: https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/xyzct_8bit__mitosis.ome.zarr 

Visualise big remote OME-Zarr data in the same way:
S3 URL: https://s3.embl.de/i2k-2020/platy-raw.ome.zarr


#### Web based viewing options
Please open Google Chrome on the BAND (for some reason this does not work with Firefox on the BAND).

Replace $USER with your user name in the following url and enter it in the Google Chrome's search bar:
https://hms-dbmi.github.io/vizarr/?source=https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/xyzct_8bit__mitosis.ome.zarr 

Optional: visualise big remote OME-Zarr data
https://hms-dbmi.github.io/vizarr/?source=https://s3.embl.de/i2k-2020/platy-raw.ome.zarr 

Optional: visualise a single well from an HCS data
https://hms-dbmi.github.io/vizarr/?source=https://s3.embl.de/eosc-future/EUOS/testdata.zarr/A/1


### Segmentation 

We can also segment remotely located OME-Zarr data without explicitly downloading it.
#### Examine the dataset that is to be segmented:
```
mc tree -d 2 s3/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
```
#### Also view the data
```
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
```

#### Segment each channel
We can use the zseg package for segmenting the data via thresholding.
```
zseg threshold -r -m otsu -c 1 -ch 0 -n otsu-c1-ch0 --colormap viridis ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
```
In this command, the `-r` flag ensures that the input path is searched at the s3 bucket. The `-m` option specifies the thresholding algorithm, which in this case is the Otsu algorithm. The `c` is a coefficient that is multiplied with the found threshold value to get the effective threshold. The `-ch` species the channel 0 for segmentation. The `-n` option specifies the name of the label path created. \

Now also segment the other channel:
```
zseg threshold -r -m otsu -c 1 -ch 1 -n otsu-c1-ch1 --colormap viridis ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
```
Note that the `-c` argument has been changed.

#### Have a look at the segmented data 
```
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
```

It is also possible to apply binary postprocessing to the segmented data.
#### Apply mathematical morphology
```
zseg postprocess -r -m binary_opening -f 1,1 -l otsu-c1-ch1 --colormap viridis ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
```
Here the `-m` specifies the postprocessing method; the `-f` determines the footprint shape. Depending on the shape of the input data, it can be 2 or 3-dimensional. The `-l` can be used to decide on the name of the label image, that is subjected to the postprocessing. 

#### Now examine the OME-Zarr data:
```
mc tree -d 2 s3/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
```
```
ome_zarr info https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
```
Also visualise the data:
```
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
``` 



