#
# Plot results using Gnuplot
#

reset

Output = 0 #(0 for Windows graphics, 1 for black and white files for LaTex, 2 for colour, 3 for graphics GIF files)

LW = 3.5
set border lw 1.4
Xscale = 1; Yscale = 0.8
set style line 1 linetype 1 linecolor rgb "blue"
set style line 2 linetype 1 linecolor rgb "red"

Dir = "C:/JF/Internet/Steady-waves/"

# Surface profile

set format x "%2.0f"
set format y "%4.1f"
if (Output==0) set xlabel "x/d"; set ylabel "eta/d"
if (Output>0) set xlabel "$x/d$" offset 0,-1.5
set ylabel "$\\dfrac{\\eta}{d}$" offset -1,0
set title "Surface profile"
set nokey
set xtics offset 0,-0.5
show ytics
#set yrange [0:1.5]
set autoscale xy
File = 'F7a'
load Dir.'SetOutput.plt'
plot "Surface.res" using 1:2 with lines ls 1;\
     pause -1 "Surface"

# Velocity profiles

uu = '$u/\sqrt{g d}$'
vv = '$v/\sqrt{g d}$'
set key top center reverse Left horizontal height 2 width +6
set format x "%4.2f"
set format y "%4.2f"
if (Output==0) set xlabel 'Velocities '.uu.' and '.vv
if (Output>0) set xlabel 'Velocities '.uu.' and '.vv offset 0,-1;
set ylabel "$\\dfrac{y}{d}$" offset -1,0
set title "Velocity profiles over half a wave"
File = 'F7b'
load Dir.'SetOutput.plt'
plot "Flowfield.res" using 2:1 title 'Horizontal '.uu with lines ls 1,\
     "Flowfield.res" using 3:1 title 'Vertical '.vv with lines ls 2;\
pause -1 "uv"

# d(Velocity)/dt profiles

uu = '${\partial u}/{\partial t} /g$'
vv = '${\partial v}/{\partial t} /g$'
set key top center reverse Left horizontal height 2 width +4
set format x "%4.2f"
set format y "%4.2f"
if (Output==0) set xlabel "Partial du/dt/g and dv/dt/g"
if (Output>0) set xlabel uu.' and '.vv offset 0,-1.5;
set title "Time derivatives of velocity over half a wave"
File = 'F7c'
load Dir.'SetOutput.plt'
plot "Flowfield.res" using 5:1 title uu with lines ls 1,\
     "Flowfield.res" using 6:1 title vv with lines ls 2;\
     pause -1 "utvt"

unset output
