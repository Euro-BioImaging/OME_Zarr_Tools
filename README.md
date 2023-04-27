# OME-Zarr course

## About

This project contains materials for the practical on "Cloud hosted image data and cloud infrastructures" on 30.09.22 as part of the [NEUBIAS EOSC-Life Bioimage Analysis in the Cloud Workshop](http://eubias.org/NEUBIAS/training-schools/neubias-academy-home/defragmentation-training-school-2022/). The materials demonstrate working with OME-Zarr data located in S3 buckets. 


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

+ **napari_viewer** environment containing the napari package along with dependencies/plugins to support OME-Zarr format.

+ **bf2raw** environment containing the bioformats2raw package, which can be used to convert images into OME-Zarr format.

+ **minio** environment containing the minio client mc, which enables interaction with s3 buckets.

+ **fiji** exectuable containing a plugin for opening the OME-Zarr format.

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

### Inspection of OME-Zarr dataset

Copy the remote OME-Zarr dataset to a local directory:

`mc mirror s3minio/ome-zarr-course/ome_zarr_data/xyzct_8bit__mitosis.zarr ~/ome_zarr_data/xyzct_8bit__mitosis.zarr`

Check what we have downloaded:

`ls -la ~/ome_zarr_data/xyzct_8bit__mitosis.zarr`

You see that the OME-Zarr image comprises several files and folders.

Inspect the `.zattrs` file, which contains some important metadata:

`cat ~/ome_zarr_data/xyzct_8bit__mitosis.zarr/.zattrs`

This tells you which axes your image data has (here: time, channel, z, y, x).
It also tells you the pixel size, via the coordinateTransformations fields.
Note that we have the coordinateTransformations three times, which means that the data is stored at three resolution levels.
Note that the time, channel and z-dimensions are not downsampled, while the scale (pixel size) for x and y is increasing.

Now we would like to know how the dimensions of the data, e.g. how many timepoints and channels.
For this we inspect the array dimensions at the hightest (0) resolution level:

`cat ~/ome_zarr_data/xyzct_8bit__mitosis.zarr/0/.zarray`

The shape field contains the dimensions, you have to map them to the axes order that you retrieved with the command above.

Let's look how the data is layed out on disc (this will also allow us to understand the chunking concept):

`tree -C -L 6 -a -v ~/ome_zarr_data/xyzct_8bit__mitosis.zarr`

We see that there is a nested folder structure with one file for each xy plane.
The nesting of the folders represents the resolution level and then the axes of the data.

(Optional) we can also gain some basic information about a remotely located OME-Zarr dataset, without downloading anything:

`ome_zarr info "https://s3.embl.de/ome-zarr-course/ome_zarr_data/xyzct_8bit__mitosis.zarr"`

Here we see that the data has 3 resolution layers and its dimensions.

Inspect a remote big image data file:

`ome_zarr info "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"`

This has larger dimensions and more resolution layers. 

### Visualisation

#### Open local OME-Zarr in napari

`napari --plugin napari-ome-zarr "~/ome_zarr_data/xyzct_8bit__mitosis.zarr"`

#### Open remote OME-Zarr in napari

`napari --plugin napari-ome-zarr "https://s3.embl.de/ome-zarr-course/ome_zarr_data/xyzct_8bit__mitosis.zarr"`

Big multi-resolution data:

`napari --plugin napari-ome-zarr "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"`

When zooming in and out you may experience some lag, because napari currently needs to wait until the higher resolution data is loaded, before it can do the rendering.

This would work just the same on your computer, if you install napari and the napari-ome-zarr plugin.

#### Open remote OME-Zarr in Fiji

`fiji`

- `[ Plugins > BigDataViewer > OME-Zarr > Open OME-Zarr from S3...]`
  - S3 URL: `https://s3.embl.de/ome-zarr-course/ome_zarr_data/xyzct_8bit__mitosis.zarr`
  - [X] Log chunk loading
- Observe the output in the console window while your are browsing around. You can see how chunks of data for fetched on demand (aka lazy-loading). This makes it possible to smoothly browse TB sized cloud hosted image data on any computer.

Big multi-resolution data:

- S3 URL: `https://s3.embl.de/i2k-2020/platy-raw.ome.zarr`

Note that in BigDataViewer the browsing is smooth also for the big data, because it continues rendering even if some data still is missing.

This would work just the same on your computer, if you install Fiji and the MoBIE update site.

#### Open remote OME-Zarr in the browser with vizarr:

- Please open Google Chrome on the BAND (for some reason this does not work with Firefox on the BAND).
- `https://hms-dbmi.github.io/vizarr/?source=https://s3.embl.de/ome-zarr-course/ome_zarr_data/xyzct_8bit__mitosis.zarr`


Note that tou can also navigate to this web address on your local computer, in fact from any computer :)

Big multi-resolution data:

`https://hms-dbmi.github.io/vizarr/?source=https://s3.embl.de/i2k-2020/platy-raw.ome.zarr`

You may experience some lag...


## Converting an image data to OME-Zarr

Here the aim is to create OME-Zarr data and write it to an S3 bucket. 

Download a directory with a TIFF image from the S3 bucket onto the BAND:

`mc mirror s3minio/ome-zarr-course/image_data/ ~/image_data`

Check what has been downloaded:

`ls ~/image_data`

Convert the TIFF image into OME-Zarr, using bioformats2raw:

`bioformats2raw --compression null --resolutions 3 --scale-format-string '%2$d' ~/image_data/xyzct_8bit__mitosis.tif ~/ome_zarr_data/xyzct_8bit__mitosis_converted.zarr`

Copy the local OME-Zarr data to a S3 bucket, using mc:

`mc mirror ~/ome_zarr_data/xyzct_8bit__mitosis_converted.zarr s3minio/ome-zarr-course/ome_zarr_data/$USER/xyzct_8bit__mitosis_converted.zarr`

Check that the data has been uploaded, using mc:

`mc ls s3minio/ome-zarr-course/ome_zarr_data/`

`mc ls s3minio/ome-zarr-course/ome_zarr_data/$USER/`

Now you can again use the above methods to visualise your remote data :-)
