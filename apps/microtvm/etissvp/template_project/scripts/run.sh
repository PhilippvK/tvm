#!/bin/sh

ETISS=${ETISS_DIR:-/work/git/prj/etiss_clint_uart/ml_on_mcu/deps/install/etiss/etiss_default/}

#. ../../venv/bin/activate

#if [ -f dBusAccess.csv ]
#then
#    rm dBusAccess.csv
#fi


( until [ -p `pwd`/.tmp/uartdevicefifoin2 ] ; do sleep 1 && echo sleep ; done ; gnome-terminal -- cat `pwd`/.tmp/uartdevicefifoin2 ) &
echo B
( until [ -p `pwd`/.tmp/uartdevicefifoout2 ] ; do test -d `pwd` && sleep 1 && echo sleep || exit ; done ; gnome-terminal -- cat `pwd`/.tmp/uartdevicefifoout2 ) &
echo "TEST" > `pwd`/log.log
(gnome-terminal -- tail -f `pwd`/log.log) &
$ETISS/examples/bare_etiss_processor/run_helper.sh "$@" 2>&1 | tee -a log.log




#if [ -f metrics.csv ]
#then
#    rm metrics.csv
#fi

ELF_FILE=$1
#MEM_INI="../../out/memsegs.ini"
# TODO: determine memory layout!
#echo "Metrics:"
#python3 $ETISS/examples/bare_etiss_processor/get_metrics.py "$ELF_FILE" -t "dBusAccess.csv" -o "metrics.csv"
echo "Done!"
#cat metrics.csv
#echo "Done2!"
