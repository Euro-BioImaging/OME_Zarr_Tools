# Create a password file

echo $ACCESSKEY:$SECRETKEY > ~/.passwd-s3fs;
chmod 600 ~/.passwd-s3fs;
# Create a mountpoint if none exists
if [ ! -d ~/s3mountpoint ];
then
	mkdir ~/s3mountpoint;
fi
# Mount the s3 bucket
s3fs -o passwd_file=~/.passwd-s3fs -o url=https://s3.embl.de/ -o use_path_request_style ome-zarr-course ~/s3mountpoint;
cd ~

# When unmounting is needed:
# fusermount -u ~/s3mountpoint

