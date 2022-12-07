#!/bin/bash

if [ $# != '3' ]; then
	echo "usage : sdss_cut_out_1min.sh rd(deg) dec(deg) picture name"
	exit 0
else
  useRA=$1
  useDEC=$2
  useOutFN=$3
fi

cd Data_download/
wget 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='$useRA'&dec='$useDEC'&scale=0.7&width=110&height=110&opt=&query=&Grid=off&PhotoObjs=off' -O $useOutFN >/dev/null 2>&1
cd ..