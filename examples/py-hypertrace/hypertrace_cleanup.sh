##################################################################
#
#  Script file to cleanup previous hypertrace results 
#
#
# e.g.
# ./hypertrace_cleanup.sh --ht_luts=luts/ --ht_output_dir=/output/
#
##################################################################
echo $PWD

for i in "$@"
do
case $i in
    -lut=*|--ht_luts=*)
    ht_luts="${i#*=}"
    shift
    ;;
    -ho=*|--ht_output_dir=*)
    ht_output_dir="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

# set defaults
ht_luts="${ht_luts:-luts}"
ht_output_dir="${ht_output_dir:-output}"

# cleanup directories
rm -rf ${ht_luts}/*
rm -rf ${ht_output_dir}/*

# cleanup logs
rm -f *.*out
