#!/bin/bash

# get script directory
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  python_script_dir="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
python_script_dir="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

trap 'echo inddt; pkill -INT -P $$' INT
trap 'echo quit; pkill -QUIT -P $$' QUIT
trap 'echo term; pkill -TERM -P $$' TERM
trap 'echo kill; pkill -KILL -P $$' KILL
trap 'echo stop; pkill -STOP -P $$' STOP
trap 'echo cont; pkill -CONT -P $$' CONT

umask 0000

unset QT_HOME
unset QT_PLUGIN_PATH

source $python_script_dir/../../sourceMuratPython.sh

cd $python_script_dir

python $python_script_dir/convertJson.py $@ &

child=$!
wait "$child"

