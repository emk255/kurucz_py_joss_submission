#!/bin/bash

#Routine to setup the linelists for faster execution of synthe.
#This needs to be re-run every time the linelist or wavelength 
#range/resolution changes

outdir="Lines_v5_PL"

if [ ! -d ${outdir} ]; then
    mkdir ${outdir}
fi

#make temporary working directory
tmpdir="tmp_${outdir}"
mkdir $tmpdir
cd $tmpdir

khome=/Users/ElliotKim/Desktop/Research/kurucz
binhome=/Users/ElliotKim/Desktop/Research/kurucz/bin

#switches to turn on/off PLs, all mols, TiO, H2O
pred=0
mol=0
tio=0
h2o=0

#Note that the second to last number below is IFPRED, 1=reads in predicted lines
#LINOUT<0 no line center opacity
${binhome}/synbeg.exe <<EOF
VAC       300.0     1800.0    300000.     0.00    0     30    .001     $pred    0
AIRorVAC  WLBEG     WLEND     RESOLU    TURBV  IFNLTE LINOUT CUTOFF        NREAD
EOF

echo ""

echo "Reading atoms"
ln -s ${khome}/lines/gfallvac.latest fort.11
${binhome}/rgfall.exe
rm -f fort.11
echo ""

#read in predicted lines
if [ $pred -eq 1 ]; then
    echo "Reading predicted lines"
    ln -s ${khome}/lines/gfpred29dec2014.bin fort.11
    ${binhome}/rpredict.exe 
    rm -f fort.11
    echo ""
fi

#read in all molecules
if [ $mol -eq 1 ]; then

    echo "Reading MgO"
    ln -s ${khome}/molecules/mgodaily.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading AlO"
    ln -s ${khome}/molecules/alopatrascu.asc fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading NaH"
    ln -s ${khome}/molecules/nah.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading VO"
    ln -s ${khome}/molecules/vo/voax.asc fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/vo/vobx.asc fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/vo/vocx.asc fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading FeH"
    ln -s ${khome}/molecules/fehfx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading H2"
    ln -s ${khome}/molecules/h2bx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/h2cx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/h2xx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading HD"
    ln -s ${khome}/molecules/hdxx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading MgH"
    ln -s ${khome}/molecules/mgh.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading NH"
    ln -s ${khome}/molecules/nhax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/nhca.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    echo "Reading CH"
    ln -s ${khome}/molecules/chjorg.dat fort.11  
    ${binhome}/rmolecasc.exe 
    rm -f fort.11 
    echo ""

    echo "Reading CN"
    ln -s ${khome}/molecules/cnax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/cnbx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/cnxx12brooke.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading C2"
    ln -s ${khome}/molecules/c2ax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/c2ba.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/c2da.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/c2ea.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading CO"
    ln -s ${khome}/molecules/coax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/coxx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading OH"
    ln -s ${khome}/molecules/ohax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/ohxx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading SiH"
    ln -s ${khome}/molecules/sihax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading SiO"
    ln -s ${khome}/molecules/sioax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/sioex.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    ln -s ${khome}/molecules/sioxx.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading CrH"
    ln -s ${khome}/molecules/crhax.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""
    
    echo "Reading CaH"
    ln -s ${khome}/molecules/cah.dat fort.11
    ${binhome}/rmolecasc.exe
    rm -f fort.11
    echo ""

    #option to not read in TiO
    if [ $tio -eq 1 ]; then
	echo "Reading TiO"
	ln -s ${khome}/molecules/tio/schwenke.bin fort.11
	ln -s ${khome}/molecules/tio/eschwenke.bin fort.48
	${binhome}/rschwenk.exe
	rm -f fort.11 fort.48
	echo ""
    fi
    
    #option to not read in H2O
    if [ $h2o -eq 1 ]; then
	echo "Reading H2O"
	ln -s ${khome}/molecules/h2o/h2ofastfix.bin fort.11
	${binhome}/rh2ofast.exe
	rm -f fort.11
	echo ""
    fi

fi

echo "wrapping up..."

#move all the outputs to the out directory
mv fort.12 ../${outdir}/tfort.12
mv fort.14 ../${outdir}/tfort.14
mv fort.19 ../${outdir}/tfort.19
mv fort.20 ../${outdir}/tfort.20
mv fort.93 ../${outdir}/tfort.93

#remove the temporary working directory
cd ../
rm -r ${tmpdir}

date
