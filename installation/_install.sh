#! /bin/bash

SCRIPTPATH=$( dirname -- ${BASH_SOURCE[0]}; );

source ~/.bashrc
mkdir -p ~/Applications;
cd ~/Applications;

# Make sure FIJI is installed and the MoBIE plugin exists 
if ! ls | grep Fiji.app &> /dev/null;
then
	wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip;
	unzip fiji-linux64.zip;
	rm fiji-linux64.zip;
	Fiji.app/ImageJ-linux64 --headless --update add-update-site MoBIE https://sites.imagej.net/MoBIE/;
	Fiji.app/ImageJ-linux64 --headless --update update;
	chmod -R a+rwx Fiji.app;
	echo 'alias fiji=$HOME/Applications/Fiji.app/ImageJ-linux64' >> ~/.bashrc;
fi;

# if miniconda3 is not in the path, add it there:
if ! echo $PATH | tr ":" "\n" | grep "conda" &> /dev/null;
then
	echo PATH="$HOME/miniconda3/bin:$PATH" >> $HOME/.bashrc;
fi;

# check if conda Miniconda3 already exists, otherwise download it
if ! ls | grep Miniconda3 &> /dev/null;
then
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh;
else
	echo "Miniconda3 is already downloaded."
fi;

# grant permission for miniconda installation file and install miniconda 
if ! command -v conda &> /dev/null; 
then
	chmod +x Miniconda3-latest-Linux-x86_64.sh;
	./Miniconda3-latest-Linux-x86_64.sh -b -u;
else
	echo "Miniconda3 is already installed."
fi;

cd ~

# Now create the environments from the yml files

source ~/.bashrc
if ! ls ~/miniconda3/envs | grep minio &> /dev/null;
then 	
	conda env create -f $SCRIPTPATH/minio_env.yml;
	echo 'alias mc=$HOME/OME_Zarr_Tools/apps/mc.sh' >> ~/.bashrc;
fi;

source ~/.bashrc
if ! ls ~/miniconda3/envs | grep bf2raw &> /dev/null;
then 	
	conda env create -f $SCRIPTPATH/bf2raw_env.yml;
	echo 'alias bioformats2raw=$HOME/OME_Zarr_Tools/apps/bioformats2raw.sh' >> ~/.bashrc;
	echo 'alias tree=$HOME/OME_Zarr_Tools/apps/tree.sh' >> ~/.bashrc
fi;

source ~/.bashrc
if ! ls ~/miniconda3/envs | grep ZarrSeg &> /dev/null;
then 	
	conda env create -f $SCRIPTPATH/ZarrSeg.yml;
	echo 'alias napari=$HOME/OME_Zarr_Tools/apps/napari.sh' >> ~/.bashrc;
	echo 'alias ome_zarr=$HOME/OME_Zarr_Tools/apps/ome_zarr.sh' >> ~/.bashrc
#	echo 'alias ome_zarr=$HOME/OME_Zarr_Tools/apps/zseg.sh' >> ~/.bashrc
fi;

source ~/.bashrc
if ! ls ~/miniconda3/envs | grep nflow &> /dev/null;
then
	conda env create -f $SCRIPTPATH/nextflow_env.yml;
	echo 'alias nextflow=$HOME/OME_Zarr_Tools/apps/nextflow.sh' >> ~/.bashrc;
fi;

source ~/.bashrc
if ! cat ~/.bashrc | grep batchonvert;
then
  echo 'alias batchconvert=$HOME/OME_Zarr_Tools/BatchConvert/batchconvert.sh' >> ~/.bashrc;
fi;
source ~/.bashrc;






