#!/bin/sh

ETISS=${ETISS_DIR:-$(pwd)/../../deps/install/etiss/etiss_default}

#. ../../venv/bin/activate

#if [ -f dBusAccess.csv ]
#then
#    rm dBusAccess.csv
#fi

$ETISS/examples/bare_etiss_processor/run_helper.sh "$@"

#if [ -f metrics.csv ]
#then
#    rm metrics.csv
#fi

ELF_FILE=$1
#MEM_INI="../../out/memsegs.ini"
# TODO: determine memory layout!
echo "Metrics:"
python3 $ETISS/examples/bare_etiss_processor/get_metrics.py "$ELF_FILE" -t "dBusAccess.csv" -o "metrics.csv"
echo "Done!"
cat metrics.csv
echo "Done2!"
