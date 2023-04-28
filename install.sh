#!/usr/bin/env bash

rel_SCRIPTPATH=$( dirname -- ${BASH_SOURCE[0]}; );
source $rel_SCRIPTPATH/utils/utils.sh

ROOT=$(abspath $rel_SCRIPTPATH);
#APPS="$ROOT/apps"

# Add root path as an environmental variable
if ! echo $PATH | tr ":" "\n" | grep "OME_Zarr_Tools" &> /dev/null;
then
	echo "export OZT=$(abspath $rel_SCRIPTPATH)" >> $HOME/.bashrc;
  source ~/.bashrc
fi;

# Add the apps folder to the path
if ! echo $PATH | tr ":" "\n" | grep "apps" &> /dev/null;
then
	echo "export PATH=$ROOT/apps:$PATH" >> $HOME/.bashrc;
  source ~/.bashrc
fi;

chmod -R 777 $ROOT;
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

# check if conda Miniconda3 already exists, otherwise download it
if ! ls | grep Miniconda3 &> /dev/null;
then
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh;
else
	echo "Miniconda3 is already downloaded."
fi;

# if miniconda3 is not in the path, add it there:
if ! echo $PATH | tr ":" "\n" | grep "conda" &> /dev/null;
then
	echo PATH="$HOME/miniconda3/bin:$PATH" >> $HOME/.bashrc;
fi;

# grant permission for miniconda envs file and install miniconda
if ! command -v conda &> /dev/null;
then
	chmod +x Miniconda3-latest-Linux-x86_64.sh;
	./Miniconda3-latest-Linux-x86_64.sh -b -u;
else
	echo "Miniconda3 is already installed."
fi;

cd ~

# Now create the environments from the yml files
if ! ls ~/miniconda3/envs | grep minio &> /dev/null;
then
	conda env create -f $ROOT/envs/minio_env.yml;
fi;

if ! ls ~/miniconda3/envs | grep bf2raw &> /dev/null;
then
	conda env create -f $ROOT/envs/bf2raw_env.yml;
fi;

if ! ls ~/miniconda3/envs | grep ZarrSeg &> /dev/null;
then
	conda env create -f $ROOT/envs/ZarrSeg.yml;
fi;

if ! ls ~/miniconda3/envs | grep nflow &> /dev/null;
then
  conda env create -f $ROOT/envs/nextflow_env.yml;
fi;

# Add batchconvert and zseg to path
if ! echo $PATH | tr ":" "\n" | grep "BatchConvert" &> /dev/null;
then
	echo "export PATH=$ROOT/BatchConvert:$PATH" >> $HOME/.bashrc;
  source ~/.bashrc
fi;


if ! echo $PATH | tr ":" "\n" | grep "ZarrSeg" &> /dev/null;
then
	echo "export PATH=$ROOT/ZarrSeg:$PATH" >> $HOME/.bashrc;
fi

source ~/.bashrc;

#### make access and secret keys universally available
if ! cat $HOME/.bashrc | grep ACCESSKEY &> /dev/null;
then
	echo ACCESSKEY=$1 >> $HOME/.bashrc;
fi;

if ! cat $HOME/.bashrc | grep SECRETKEY &> /dev/null;
then
	echo SECRETKEY=$2 >> $HOME/.bashrc;
fi;

source $HOME/.bashrc;

### configure mc
chmod -R a+rwx $ROOT/apps;
mc alias set s3minio https://s3.embl.de $ACCESSKEY $SECRETKEY;

source $HOME/.bashrc;

### Make sure the correct python is used in the batchconvert script
v_info=$( python --version )
VP=${v_info:7:1}

if [[ $VP == 3 ]];
  then
    printf "The following python will be used to execute python commands in batchconvert script: $( which python ) \n"
    if ! [ -f $ROOT/BatchConvert/pythonexe ];then
	    ln -s $( which python ) $ROOT/BatchConvert/pythonexe;
    fi
elif ! [[ $VP == 3 ]];
  then
    printf "Python command refers to the following python: $( which python ), which cannot be used in the batchconvert script \nWill search the system for python3 \n";
    if command -v python3 &> /dev/null;
      then
	      printf "python3 was found at $( which python3 ) \n";
	      printf "This python will be used in the batchconvert script \n";
        if ! [ -f $ROOT/BatchConvert/pythonexe ];then
	        ln -s $( which python3 ) $ROOT/..BatchConvert/pythonexe;
        fi
      else
        printf "Looks like python3 does not exist on your system or is not on the path. Please make sure python3 exists and on the path. \n"
        exit
    fi
fi
# configure batchconvert s3
batchconvert configure_s3_remote --remote s3minio --url https://s3.embl.de --access $ACCESSKEY --secret $SECRETKEY --bucket ome-zarr-course
# configure zseg s3
zseg configure_s3_remote --url s3.embl.de --access $ACCESSKEY --secret $SECRETKEY --region eu-west-2




