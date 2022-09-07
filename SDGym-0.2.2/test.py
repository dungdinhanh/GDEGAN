
from sdgym.synthesizers.ctgan import CTGANSynthesizer
import sdgym

dataset_name="adult"
cache_dir="temp"
outpath="temp/abc.csv"
import os
os.makedirs(cache_dir, exist_ok=True)

model = CTGANSynthesizer
scores = sdgym.run(synthesizers=model, datasets=[dataset_name],cache_dir=cache_dir, output_path=outpath, iterations=3)