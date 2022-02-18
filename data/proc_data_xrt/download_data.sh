
dataset=$1
if [ ${dataset} != "ogbn-arxiv" ] && [ ${dataset} != "ogbn-products" ] && [ ${dataset} != "ogbn-papers100M" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

# Download if not yet exist
if [ -d ${dataset} ]; then
  echo "dataset=${dataset} available. Skip downloading."
else
  wget https://archive.org/download/pecos-dataset/giant-xrt/${dataset}.tar.gz
  tar -zxvf ${dataset}.tar.gz
fi
