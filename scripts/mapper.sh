#!/bin/bash

unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
python_=python3
else python_=python
fi

installs() {
    if [[ "$unamestr" == 'Linux' ]]; then
        lsb_release -a
        apt-get update
    fi
}

package_name_agg=wos_agg
package_name_gg=graph_tools
data_path=../
year=$1
maxlist=$2
log=../wos_ef_$year.log
verb=INFO

setup_data() {
    tarfile=`ls *tar.gz`
    echo "Found tar.gz files" $tarfile "of size" $(du -smh $tarfile | awk '{print $1}')
    for tf in $tarfile; do
        tar xf $tf
    done
}

clone_repo() {
echo "starting cloning $1"
git clone https://github.com/alexander-belikov/$1.git
echo "*** list files in "$1":"
ls -lht ./$1
echo "*** list files in "$1"/"$1 ":"
ls -lht ./$1/$1
cd ./$1
echo "starting installing $1"
$python_ ./setup.py install
cd ..
}

exec_driver() {
cd ./$1
echo "starting exec_driver $1"
echo $python_
$python_ ./driver.py -s $data_path -d $data_path -y $year -l $log -v $verb -m $maxlist
cd ..
}

post_processing() {
echo "*** all files sizes :"
ls -thor *
}

# Install packages
# installs
# Setup the data files
setup_data
# Clone repos from gh
clone_repo $package_name_agg
clone_repo $package_name_gg
# Execute the driver script
exec_driver $package_name_agg
# Prepare the results
post_processing
