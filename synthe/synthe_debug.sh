#!/bin/bash

date

khome=/Users/ElliotKim/Desktop/Research/kurucz
krun=/Users/ElliotKim/Desktop/Research/kurucz
bindir=/Users/ElliotKim/Desktop/Research/kurucz/bin

indir="${khome}/grids/$1/atm/"
outdir="${khome}/grids/$1/spec/"
moldir="${khome}/grids/$1/molnden/"

echo "Output Dir: ${outdir}"

arr=(`ls $indir`)

echo "Moving into temporary working directory...."
rm -rf ${krun}/synthe/stmp_$1
mkdir ${krun}/synthe/stmp_$1
cd ${krun}/synthe/stmp_$1

model=${arr[0]}
linedir="Lines_v5_PL"

asc="${model/atm/spec}"
asc="${asc/dat/spec}"

echo $linedir

#generate synthe-ready input file
$bindir/at12tosyn.exe $indir$model $model

#re-link files each time
cp ${khome}/lines/molecules.dat fort.2
cp ${khome}/lines/continua.dat  fort.17

# Capture xnfpelsyn debug output (includes TK debug statements)
$bindir/xnfpelsyn.exe < $model  > xnfpelsyn_debug.log 2>&1

#link the input files generated from synthe.setup
cp ${khome}/synthe/${linedir}/tfort.12 fort.12
cp ${khome}/synthe/${linedir}/tfort.14 fort.14
cp ${khome}/synthe/${linedir}/tfort.19 fort.19
cp ${khome}/synthe/${linedir}/tfort.20 fort.20
cp ${khome}/synthe/${linedir}/tfort.93 fort.93

cp ${khome}/lines/he1tables.dat fort.18

#run synthe, the main program
echo "running synthe...."
# Capture stdout/stderr (unit 6) so XLINOP debug prints are retained
stdbuf -oL -eL $bindir/synthe.exe 2>&1 | tee synthe_debug.log

ln -s $model fort.5
ln -s $khome/infiles/spectrv_std.input fort.25
    
echo "running spectrv..."
# Capture spectrv.exe output to debug log
$bindir/spectrv.exe 2>&1 | tee spectrv_debug.log

#save the molecular number density profiles
# rm -f fort.2 fort.5 fort.25 fort.33

#convert the spectrum into an ascii file
mv fort.7 fort.1

echo "running syntoascanga..."
# Remove lineinfo.dat if it exists (Fortran won't overwrite existing files)
rm -f lineinfo.dat
rm -f headinfo.dat
$bindir/syntoascanga.exe  > /dev/null
mv specfile.dat $outdir$asc

#clean up
# rm -f fort.*

echo "removing working directory"
cd ../
# rm -r stmp_$1

date
