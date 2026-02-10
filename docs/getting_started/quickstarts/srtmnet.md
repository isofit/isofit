# Quick Start with sRTMnet (Recommended for new users)

sRTMnet is an emulator for MODTRAN 6, that works by coupling a neural network with a surrogate RTM (6S v2.1).


## Automatic (Recommended)

ISOFIT can automatically install 6S and sRTMnet with the latest versions:

```
$ isofit download sixs
$ isofit download srtmnet
```

The above commands will ensure these models are built and available for ISOFIT.

<blockquote style="border-left: 5px solid lightblue; padding: 0.5em 1em; margin: 1em 0;" markdown="1">

:information_source: A commonly useful option `-b [path]`, `--base [path]` will set the download location for all products:
```
$ isofit -b extra-downloads/ download all
```

This will change the download directory from the default `~` to `./extra-downloads/`

See [data](../../extra_downloads/data.md) for more information.

</blockquote>

## Manual (Advanced)

The following procedure walks through the steps required to install sRTMnet manually:

1. Download [6S v2.1](https://salsa.umd.edu/files/6S/6sV2.1.tar), and compile. If you use a modern system, it is likely you will need to specify a legacy compiling configuration by changing line 3 of the Makefile to:
```
EXTRA = -O -ffixed-line-length-132 -std=legacy
```

2. Configure your environment by pointing the SIXS_DIR variable to point to your installation directory.

3. Download the [pre-trained sRTMnet neural network](https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/sRTMnet_v120.h5), as well as some [auxiliary data](https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/sRTMnet_v120_aux.npz). This will give you an hdf5 and an aux file. It is important that you store both in the same directory. You will likely need to set the path to 6S and sRTMnet for the ISOFIT ini file as well as rebuild the examples.
To do this, execute:
```
$ isofit --path sixs /path/to/sixs/ --path srtmnet /path/to/sRTMnet/ build
```

4. Run one of the following examples:

```
    # Small example pixel-by-pixel
    $ cd $(isofit path examples)/image_cube/small/
    $ ./default.sh
```
```
    # Medium example with empirical line solution
    $ cd $(isofit path examples)/image_cube/medium/
    $ ./empirical.sh
```
```
    # Medium example with analytical line solution
    $ cd $(isofit path examples)/image_cube/medium/
    $ ./analytical.sh
```
