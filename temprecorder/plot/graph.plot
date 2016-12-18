set terminal png size 10000,2000 enhanced font "Helvetica,10"
set output 'temperature.png'
set xdata time
set timefmt "%Y-%m-%d %H:%M:%S"
set format x "%Y-%m-%d %H:%M"
set xtics 7200
set ytics 1
set grid ytics xtics
set style line 1 linewidth 2 linecolor rgb "red"
set style line 2 linewidth 2 linecolor rgb "blue"
#plot for [file in `ls records/rec*`] file u 1:3 w l
plot '<(cat /www/temprecorder/records/rec_*)' using 1:3 title 'Inside' with lines linestyle 1, \
    '<(cat /www/temprecorder/records/rec_*)' using 1:4 title 'Outside' with lines linestyle 2
