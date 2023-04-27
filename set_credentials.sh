if ! cat $HOME/.bashrc | grep ACCESSKEY &> /dev/null;
then
	echo ACCESSKEY=$1 >> $HOME/.bashrc;
fi;

if ! cat $HOME/.bashrc | grep SECRETKEY &> /dev/null;
then
	echo SECRETKEY=$2 >> $HOME/.bashrc;
fi;

source $HOME/.bashrc;


