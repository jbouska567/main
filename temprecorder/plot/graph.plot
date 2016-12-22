set terminal png size 10000,5000 enhanced font "Helvetica,10"
set output 'temperature.png'
set xdata time
set timefmt "%Y-%m-%d %H:%M:%S"
set format x "%Y-%m-%d %H:%M"

# styly vykreslovani
# vnitrni a venkovni teplota
set style line 1 linewidth 2 linecolor rgb "red"
set style line 2 linewidth 2 linecolor rgb "blue"
# mrizka mtics tics
set style line 100 lt 1 lc rgb "gray" lw 2
set style line 101 lt 0.5 lc rgb "gray" lw 1

# xtics je v sekundach (jelikoz je xdata time)
set xtics 7200
# rozsah pro vnitrni teplotu
set ytics 1
set yrange [0:30]
# rozsah pro venkovni teplotu
set y2tics 1
set y2range [-15:15]

# mrizka
set grid ytics y2tics ls 101, ls 101
set grid mxtics xtics ls 100, ls 101

# horizontalni cary s teplotou 0 a 20
# (graph je x pozice v grafu 0-1)
# (first prvni sada souradnic x, y)
# (second je druha sada souradnic x2, y2)
set arrow 1 from graph 0, first 20 to graph 1, first 20 nohead ls 1
set arrow 2 from graph 0, second 0 to graph 1, second 0 nohead ls 2


# vykresleni dat
#plot for [file in `ls records/rec*`] file u 1:3 w l
plot '<(cat /www/temprecorder/records/rec_*)' using 1:3 title 'Inside' with lines linestyle 1, \
    '<(cat /www/temprecorder/records/rec_*)' using 1:4 title 'Outside' with lines linestyle 2
