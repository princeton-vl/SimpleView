wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
key="$1"
case $key in
	pretrained)
		wgetgdrive 1SYwkAbahftSEm3ykHK-SdBrDi7hwa2BC tmp/pretrained.zip
		unzip -o tmp/pretrained.zip
    		;;
	modelnet40)
		wget --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
		unzip modelnet40_ply_hdf5_2048.zip
		mv modelnet40_ply_hdf5_2048 data
		rm -r modelnet40_ply_hdf5_2048.zip
    		wgetgdrive 14Xcr8kG_1VFMpxpklH96U3d78k7lTuCq tmp/modelnet40_ply_hdf5_2048_valid_small.zip
		unzip -o tmp/modelnet40_ply_hdf5_2048_valid_small.zip
		mv modelnet40_ply_hdf5_2048_valid_small/* data/modelnet40_ply_hdf5_2048/
		rm -r modelnet40_ply_hdf5_2048_valid_small
		;;
    	*)
    		echo "unknow argument $1" # unknown argument
    		;;
esac
rm -r tmp
