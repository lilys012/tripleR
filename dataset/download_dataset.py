import os
import pathlib
from beir import util

def main():
    
    out_dir = pathlib.Path(__file__).parent.absolute()
    
    dataset_files = [ "nfcorpus.zip",
                     "fiqa.zip",  "fever.zip",
                      "scifact.zip"]
    
    for dataset in dataset_files:
        
        zip_file = os.path.join(out_dir, dataset)
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
        
        print("Downloading {} ...".format(dataset))
        util.download_url(url, zip_file)
        
        print("Unzipping {} ...".format(dataset))
        util.unzip(zip_file, out_dir)

if __name__ == '__main__':
    main()