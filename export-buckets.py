import argparse
from connector.store import ImageStore, AspectBucket
import pickle

parser = argparse.ArgumentParser(description="Waifu Diffusion Bucket Exporter")
parser.add_argument(
    "--export_path",
    type=str,
    default=None,
    required=True,
    help="The path to use for exporting."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    required=True,
    help="The batch size to use."
)
args = parser.parse_args()

print('creating image store...')
image_store = ImageStore()
print('creating aspect buckets...')
bucket = AspectBucket(image_store, args.batch_size)
print('writing aspect buckets to file...')
with open(args.export_path, 'wb') as f:
    pickle.dump(bucket, f)

print('done.')
