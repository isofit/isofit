# Additional Data

To get started with ISOFIT examples, simply execute the two following commands:

```
$ isofit download all
$ isofit build
```

The first will download all additional ISOFIT dependencies and configure them for the current system. The second will build the ISOFIT examples using the configured dependencies. From there, examples will be available under `~/.isofit/examples/`. Each subdirectory will have one or more scripts that are prepared for execution.

???+ note

    Commonly useful options `-b [path]`, `--base [path]` will set the download location for all products:

    ```
    $ isofit -b extra-downloads/ download all

    This will change the download directory from the default
    ```

???+ warning

    If on MacOS, executing the `make` command may fail if the user hasn't agreed to the Xcode and Apple SDKs license yet. In these cases, it may be required to run the following command in order to compile the programs that use it (6S, LibRadTran):
    ```
    $ sudo xcodebuild -license
    ```

If there are any issues, please report them to the [ISOFIT repository](https://github.com/isofit/isofit/issues).

The contents below go into further details about additional commands.

## Configuration Options

ISOFIT uses INI files to configure the location of extra dependencies that are not included in the default ISOFIT installation. These include things like larger data files and the ISOFIT examples.

<blockquote style="border-left: 5px solid lightblue; padding: 0.5em 1em; margin: 1em 0;" markdown="1">

:information_source: The below commands assume a user is in their home directory, aka `~`. For Mac, this is commonly `/Users/[username]/`. The examples on this page will use `~`, but in practice this path and other relative paths will be automatically replaced with the absolute path.

</blockquote>

When the `isofit` command is first executed, it will create a directory under the user's home directory named `.isofit` as well as initialize a default `isofit.ini` file:

```
$ isofit
Wrote to file: ~/.isofit/isofit.ini

$ cat ~/.isofit/isofit.ini
[DEFAULT]
data = ~/.isofit/data
examples = ~/.isofit/examples
imagecube = ~/.isofit/imagecube
srtmnet = ~/.isofit/srtmnet
sixs = ~/.isofit/sixs
modtran = ~/.isofit/modtran
```

Notice the default location for all paths is `~/.isofit/`. These can be modified by either directly editing the INI file or by using the ISOFIT CLI:

```
$ isofit --help
Usage: isofit [OPTIONS] COMMAND [ARGS]...

  ISOFIT contains a set of routines and utilities for fitting surface,
  atmosphere and instrument models to imaging spectrometer data.

  Repository: https://github.com/isofit/isofit
  Documentation: https://isofit.readthedocs.io/en/latest
  Report an issue: https://github.com/isofit/isofit/issues

Options:
  -i, --ini TEXT          Override path to an isofit.ini file
  -b, --base TEXT         Override the base directory for all products
  -s, --section TEXT      Switches which section of the ini to use
  -p, --path TEXT...      Override paths with the format `-p [key] [value]`
  -k, --keys TEXT...      Override keys with the format `-k [key] [value]`
  --save / -S, --no-save  Save the ini file
  --preview               Prints the environment that will be used. This
                          disables saving
  --version               Print the installed ISOFIT version
  --help                  Show this message and exit
```

Using a data override flag (`--path [name] [path]`) will update the the INI with the provided path:

```
$ isofit -p examples tutorials
Wrote to file: ~/.isofit/isofit.ini

$ isofit download paths
Download paths will default to:
- data = ~/.isofit/data
- examples = ~/tutorials
- imagecube = ~/.isofit/imagecube
- srtmnet = ~/.isofit/srtmnet
- sixs = ~/.isofit/sixs
- modtran = ~/.isofit/modtran
```

For advanced users, the INI file itself as well as the base directory and the section of the INI may be modified:

```
$ isofit -i test.ini -b test -s test -p data test
Wrote to file: test.ini

$ cat test.ini
[DEFAULT]
data = ~/.isofit/data
examples = ~/tutorials
imagecube = ~/.isofit/imagecube
srtmnet = ~/.isofit/srtmnet
sixs = ~/.isofit/sixs
modtran = ~/.isofit/modtran

[test]
data = ~/dev/test
examples = ~/dev/test/examples
imagecube = ~/dev/test/imagecube
srtmnet = ~/dev/test/srtmnet
sixs = ~/dev/test/sixs
modtran = ~/dev/test/modtran
```

The `DEFAULT` section is still instantiated, but now there's a `test` section with a different `data` path than the default. Also note the default `examples` is different -- this is because the above examples changed it in the default INI, which is still read if available.

Additionally, these paths may be used in command-line arguments via the `isofit path` command. For example:

```
$ cd $(isofit path examples)
$ ls $(isofit path data)/reflectance
$ cd $(isofit -i test.ini -s test path srtmnet)
```

## Downloads

ISOFIT comes with a `download` command that provides users the ability to download and install extra files such as larger data files and examples. To get started, execute the `isofit download --help` in a terminal. At this time, there are 9 subcommands:

| Command     | Description |
|-------------|-------------|
| `paths`     | Displays the currently configured path for a download |
| `all`       | Executes all of the download commands below |
| `data`      | Downloads ISOFIT data files from https://github.com/isofit/isofit-data |
| `examples`  | Downloads the ISOFIT examples from https://github.com/isofit/isofit-tutorials |
| `imagecube` | Downloads required data for the image_cube example |
| `sRTMnet`   | Downloads the sRTMnet model |
| `sixs`      | Downloads and builds 6sv-2.1 |
| `plots`     | Downloads and installs the ISOFIT plots package from https://github.com/isofit/isofit-plots |
| `libradtran`| Downloads and installs the LibRadTran radiative transfer model. This is experimental and not guaranteed to work. For advanced users only |

The paths for each download are defined in the currently active INI.
Download paths can be modified by either directly modifying the `~/.isofit/isofit.ini` or by using `isofit --help` flags (shown above).
Additionally, download paths may be temporarily overridden and not saved to the active INI by providing a `--output [path]`. For example:

```
$ isofit download data --help
Usage: isofit download data [OPTIONS]

Downloads the extra ISOFIT data files from the repository
https://github.com/isofit/isofit-data.

Run `isofit download paths` to see default path locations.
There are two ways to specify output directory:
  - `isofit --data /path/data download data`: Override the ini file. This will save the provided path for future reference.
  - `isofit download data --path /path/data`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
It is recommended to use the first style so the download path is remembered in the future.

Options:
-p, --path TEXT  Root directory to download data files to, ie. [path]/data
-t, --tag TEXT   Release tag to pull  [default: latest]
--overwrite      Overwrite any existing installation
-c, --check      Only check for updates
--help           Show this message and exit.
```

Some subcommands have additional flags to further tweak the download, such as `data` and `examples` having a `--tag` to download specific tag releases, or `sRTMnet` having `--version` for different model versions, but it is recommended to use the default to pull the most up-to-date download for each.
