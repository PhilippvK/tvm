#!/bin/sh

ETISS=${ETISS_DIR:-/work/git/prj/etiss_freertos/etiss-public-fork/build/installed/}

#. ../../venv/bin/activate

#if [ -f dBusAccess.csv ]
#then
#    rm dBusAccess.csv
#fi

echo "<> Setting up ETISSVP..."
#echo -e "\n\n\r===================FIFO IN=========================\n" >> /tmp/etissvp_rx.log
#( until [ -p `pwd`/.tmp/uartdevicefifoin2 ] ; do sleep 1 ; done ; cat `pwd`/.tmp/uartdevicefifoin2 >> /tmp/etissvp_rx.log) &
#echo -e "\n\n\r===================FIFO OUT=========================\n" >> /tmp/etissvp_tx.log
#( until [ -p `pwd`/.tmp/uartdevicefifoout2 ] ; do test -d `pwd` && sleep 1 || exit ; done ; cat `pwd`/.tmp/uartdevicefifoout2 >> /tmp/etissvp_tx.log) &
#echo "Monitors for the virtual uart have been set up. Use 'tail -f /tmp/etissvp_rx.log' and 'tail -f /tmp/etissvp_tx.log' to inspect them."
echo -e "\n\n\r===================ETISS=========================\n" >> /tmp/etissvp.log
echo "Simulation logs will be available in /tmp/etissvp.log!"
$ETISS/examples/bare_etiss_processor/run_helper.sh "$@" 2>&1 | tee -a /tmp/etissvp.log

#if [ -f metrics.csv ]
#then
#    rm metrics.csv
#fi

ELF_FILE=$1
#MEM_INI="../../out/memsegs.ini"
# TODO: determine memory layout!
#echo "Metrics:"
#python3 $ETISS/examples/bare_etiss_processor/get_metrics.py "$ELF_FILE" -t "dBusAccess.csv" -o "metrics.csv"
#cat metrics.csv
#echo "Done2!"

echo "<> Done!"
