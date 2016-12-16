set terminal png size 10000,2000 enhanced font "Helvetica,10"
set output 'temperature.png'
set xdata time
set timefmt "%Y-%m-%d %H:%M:%S"
set format x "%Y-%m-%d %H:%M"
set xtics 7200
set grid ytics xtics
#plot for [file in `ls records/rec*`] file u 1:3 w l
plot '<(cat records/rec*)' u 1:3 w l
