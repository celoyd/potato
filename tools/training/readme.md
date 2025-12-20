## Training

Training itself is done with `train.py`, but setting up the data for it to use is a relatively involved process, called chipping, which takes up most of this tutorial.

We will use [`aws-cli`](https://github.com/aws/aws-cli). In principle it’s all possible using the HTTPS API endpoints, but `aws s3 sync` makes things simpler.

### Source data and directory setup

Here we will use [the Maxar Open Data Program’s imagery for the Emilia-Romagna flooding of 2023](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/Emilia-Romagna-Italy-flooding-may23/collection.json?.language=en). This is a 51 gigabyte download (some of which we won’t end up using); if this is prohibitive, pick a subset of it or another data source. Welcome to remote sensing and its inconveniently large data.

We’ll need two folders, each of which will hold a lot of TIFFs. I do i/o-heavy tasks like this on low-end external solid-state drives, with symlinks to keep the paths convenient. But this is a quickstart, so the simple way is:


```bash
mkdir -p ards/italy32
mkdir -p chips/italy32
```

Now we can fetch the data:

```bash
aws s3 sync s3://maxar-opendata/events/Emilia-Romagna-Italy-flooding-may23/ard/ ards/italy23
```

## Chipping

A _chip_ is jargon for a small image pulled out of a larger image.

The chipping process copies chips out of collections of images and resamples them into training pairs. The main arguments to the chipper are the path of an ARD (the tiled delivery package for an image) and the path of a directory in which to put training pairs of chips (as [`.pt` files](https://pytorch.org/docs/stable/generated/torch.save.html)). We also keep a log file and request, in this case, 1024 chips:

```bash
python chip.py make-chips --ard-dir ards/italy23 --chip-dir chips/italy23 --log italy23.log -n 1024
```

The chipper is clever enough not to chip images from CIDs (catalog IDs, or image strips) that are not from WorldView-2 or -3, so, assuming correct ARD input, it only produces chips that are technically compatible with Potato. It also filters //

## More advanced topics

In this section we visit some techniques that may help with training, but are not necessary to get started, and link to their in-depth documentation.

### CID allow-lists

You may want to limit which CIDs (catalog IDs, or image strips) you use to make training data. The chipper itself only knows enough to skip satellites that aren't WorldView-2 or -3 (the ones whose data Potato was trained on).

For example, as described in the documentation on band misalignment, I prefer to train on cloud-free CIDs with little surface water. To support this, I spent a day early this summer subjectively evaluating every image in the Maxar Open Data Program on axes of cloudiness, surface water coverage, and interestingness of landcover. These are weighted into an overall quality index, and I selected only the top-scoring CIDs. There are plenty of other things you might want to do with an allow-list; for example, you might want to select individual scenes for train/test splits.

TK link to ARD CIDs sheet

The `--allow-list` option expects a path that’s a plaintext file with one CID per line. There’s no extra parsing; it won’t recognize a regex, for example.

### Restarting

For some development strategies, it’s handy to chip a few hundred or a few thousand at a time. The chip selection within source images is diffused with [a deterministic low-discrepancy sequence](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) that (to simplify) does a reasonable job of maximizing distance between chip centerpoints for any given number of chips. It’s thus slightly better at information-per-chip efficiency than random selection would be. However, since it’s deterministic, if we ran the chipper twice with the same arguments, we’d get the same chips. So there’s an `-s` or `--starting-from` argument that picks up from some point. Chip numbering is 0-based, so you would run with, say, `--count 1024` and then pick up with `--starting-from 1024`.

### Linking (storage load balancing)

If you have more than one folder of chips, it’s nice to be able to mix them into a single folder. The `link-chips` chipper command does this by taking a set of source directories and making symlinks to their chips in the destination directory. The links are renamed (with the same <var>integer</var>.pt naming scheme) and deterministically shuffled so that a `DataLoader` gets a mixed sample even if it only reads the first _n_ chips. This is convenient to change the mix of source data for training, and to load-balance chip reads across multiple storage devices.

### Synthetic chips

TK

## Training

To start training from the pretrained weights shipped in the repo, we can use:

TK fix code + change training script

```bash
python train.py --load-epoch 63 --lr 1e-4 --chips chips7 --epoch-length 10240 --epochs 100 --test-chips chips7 --workers 5

python train.py --session spaceheater --chips chips --test-chips chips --epoch-length 1024 --epochs 10 --lr 1e-4
```

To train from scratch:

```bash
python train.py --session demo --chips chips --test-chips chips --epoch-length 1024 --epochs 10 --lr 1e-4
```

Using the same chips for training and testing is a bad idea for serious training sessions (because it makes the test loss uninformative); you will want to set up a separate testing chip directory.

Once the first epoch is done, you can use `demo.py` with your first checkpoint. Also note that the training script logs for tensorboardX, although I have no idea why you would want that.
