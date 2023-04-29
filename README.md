# OME-Zarr course

## About

This project contains materials for the practical on "Cloud hosted image data and cloud infrastructures" on 08.04.23 as part of the [NEUBIAS EOSC-Life Bioimage Analysis in the Cloud Workshop](http://eubias.org/NEUBIAS/training-schools/neubias-academy-home/defragmentation-training-school-2022/). The materials demonstrate working with OME-Zarr data located in S3 buckets. 


## Connect to the BAND cloud computer

In this practical we are using the [BAND](https://band.embl.de/#/eosc-landingpage) cloud computing.


#### Connect the first time

To connect to the BAND, please follow these steps:

- Go to the [BAND](https://band.embl.de/#/eosc-landingpage) web site
- Read the [user guide](https://docs.google.com/document/d/1TZBUsNIciGMH_g4aFj2Lu_upISxh5TV9FBMrvNDWmc8/edit?usp=sharing)
- Accept the terms of usage, to activate the login button
- `[ Login ]`
  - Please use your Google account
- Choose 4 CPU and 8 GB memory
- `[ Launch ]`
- On the same page below, now `[ Go to Desktop ]`

#### Re-connect to a session

If you did not stop the recent session you can simply

- Go to the [BAND](https://band.embl.de/#/eosc-landingpage) web site
- `[ Go to Desktop ]`

## Software installation

For this practical we need several software to be installed:

+ **bf2raw** environment containing the bioformats2raw package, which can be used to convert images into OME-Zarr format.

+ **minio** environment containing the minio client mc, which enables interaction with s3 buckets.

+ **nflow** environment containing the workflow management system nextflow.

+ **ZarrSeg** environment containing the napari package along with dependencies/plugins to support OME-Zarr format
and our command line tool, ZarrSeg, for segmentation of OME-Zarr data.

+ **fiji** exectuable containing a plugin for opening the OME-Zarr format.

+ **batchconvert** executable for parallelised conversion of image data and file transfer between the local filesystem and the s3 buckets.

To install the software you will need to launch your BAND cloud computer (see above) and use Firefox and the terminal window.

![Image](docs/BAND_Terminal_Firefox.png)


Please follow those steps:

1. Launch the BAND cloud computer (see instructions above)
1. Open a Terminal window (see screenshot above)
1. Open Firefox (see screenshot above) on the BAND.
1. In Firefox browse to the Google Doc that has been shared with you (it is the same Google Doc that brought you to this page; you need to type the address starting with `tinyurl...` into the Firefox search bar and press enter)
1. Copy the command (starting with `cd ~ ...`) from the Google Doc into the Terminal window and press enter
1. This can take about 10 minutes.
1. It should finish saying: `Added s3minio successfully.`

## Practical

In general, everything that is formatted like this `command bla bla` should be copy and pasted into the BAND terminal window and enter should be pressed to execute that code.

Although we are doing this practical on a cloud computer, you should be able to reproduce all of this on a local (Linux) computer. Here [are the installation instructions](https://git.embl.de/oezdemir/course_scripts/-/tree/main/installation), which you may have to modify a bit, but the conda installation parts should work on any system. Let us know if you need help with that!

#### Inspection of the remote datasets

Check out what we have at our s3 bucket:
``` 
mc tree s3minio/ome-zarr-course/
mc ls s3minio/ome-zarr-course/data/MFF/
mc ls s3minio/ome-zarr-course/data/JPEG/
```

#### Conversion of the remote datasets

The remote datasets can be converted in a parallelised manner by using the `batchconvert` tool. 
```
batchconvert omezarr -st s3 -dt s3 --drop_series data/MFF data/ZARR/$USER;
```
This command will map each input file to a single OME-Zarr series. 

**Grouped conversion mode:**
```
batchconvert omezarr -st s3 -dt s3 --drop_series --merge_files --concatenation_order t data/JPEG data/ZARR/$USER;
```
### Check what has changed at the s3 end:
mc tree -d 2 s3minio/ome-zarr-course/

### Copy the Zarr data to the home folder
mc mirror s3minio/ome-zarr-course/data/ZARR/$USER ~/data/ZARR;

### Visualise locally and remotely with napari
napari --plugin napari-ome-zarr ~/data/ZARR/xyzct_8bit__mitosis.ome.zarr
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/xyzct_8bit__mitosis.ome.zarr

### Visualise locally with fiji
fiji ;

### Other viewing options
https://kitware.github.io/itk-vtk-viewer/app/?fileToLoad=https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/xyz_8bit__nucleus.ome.zarr
# add vizarr and neuroglancer


### REMOTE SEGMENTATION 
# Have a look at the Zarr data before segmenting
mc tree -d 2 s3minio/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
# Segment both channels
zseg threshold -r -m otsu -c 1 -ch 0 -n otsu-c1-ch0 ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
zseg threshold -r -m otsu -c 1 -ch 1 -n otsu-c1-ch1 ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
# Have a look after segmenting
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
# Postprocess via mathematical morphology
zseg postprocess -r -m binary_opening -f 1,1 -l otsu-c1-ch1 ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
#zseg postprocess -r -m binary_opening -f 2,2 -l otsu-c1-ch1 ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;
# Have a final look
mc tree -d 2 s3minio/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
ome_zarr info mc tree -d 2 s3minio/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr
napari --plugin napari-ome-zarr https://s3.embl.de/ome-zarr-course/data/ZARR/$USER/23052022_D3_0002_positiveCTRL.ome.zarr;





