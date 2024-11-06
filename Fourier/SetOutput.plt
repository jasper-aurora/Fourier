# Subroutine file - to set terminal and output files
# You need values for Output, Xscale, Yscale, and File

if (Output==0) {
	set size Xscale,Yscale;\
	Margins = -1;\
	set terminal qt;
	}

if (Output > 0) {
	set size Xscale,Yscale;\
	Margins = 2;
	}

if (Output==1) {
	set terminal epslatex blacktext font '"",6';\
	set output File.'.eps';
	}

if (Output==2) {
	set terminal epslatex color colortext font '"",6';\
	set output File.'.eps';
	}

if (Output==3) {
	set terminal gif font 'times' 12 size 800,600;\
	set size 1,1;\
	set output File.'.gif';\
	Margins = -1;
	}

if (Output==4) {
	set terminal jpeg font 'times' 10;\
	set output File.'.jpg';\
	Margins = -1;
	}

set bmargin Margins
set lmargin Margins
set rmargin Margins
set tmargin Margins
