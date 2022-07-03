#!/bin/sh

export PATH=/Applications/MATLAB_R2021b.app/bin:$PATH

echo "Enter filename with fluorescence calcium transients (without '.txt'): "
echo "[Make sure each row corresponds to one neuron]"
read filename
echo "File name is ${filename}.txt"

echo "Enter sampling frequency (in Hz):"
read sf
echo "Frequency = $sf Hz."

echo "remove motion artifact ..."

#matlab -nodisplay -nodesktop -r "try remove_motion_artifact('$filename'); catch; end; quit"

#echo "z-score fluorescence ..."

#matlab -nodisplay -nodesktop -r "try zscore_f('$filename'); catch; end; quit"

#echo "smooth the signal with total-variation regularization ..."

#matlab -nodesktop -r "try tvreg_smoothen('$filename', $sf); catch; end; quit"

echo "Now you see the noise correlation, is it large enough that you want to use the smoothened fluorescence, instead of the original fluorescence? (y/n)"

read yn

case $yn in
    [yY] ) echo "replace signal with the smooth version ...";
	   filename="${filename}_smooth";;
    [nN] ) echo "keep the original signal ...";;
    * ) echo "invalid response";;
esac

echo "Filename is ${filename}.txt"
	   
echo "Now we compute the granger causality..."
#echo "enter lags: "
#read lags
#echo "lags = ${lags}"
#echo "enter p-value threshold:"
#read pthres
#echo "p-value threshold = ${pthres}"

./compute_gc.py
