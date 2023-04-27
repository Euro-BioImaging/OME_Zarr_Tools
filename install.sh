#!/usr/bin/env bash

SCRIPTPATH=$( dirname -- ${BASH_SOURCE[0]}; )
source $SCRIPTPATH/installation/_install.sh;
source ~/.bashrc;
SCRIPTPATH=$( dirname -- ${BASH_SOURCE[0]}; );
chmod -R a+rwx $SCRIPTPATH/apps;
mc alias set s3minio https://s3.embl.de $ACCESSKEY $SECRETKEY;

