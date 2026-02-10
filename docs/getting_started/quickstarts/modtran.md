# Quick Start using MODTRAN 6.0

This quick start presumes that you have an installation of the MODTRAN 6.0 radiative transfer model. This is the preferred radiative transfer option if available, though we have also included interfaces to the open source LibRadTran RT code as well as to neural network emulators.

1. Create an environment variable MODTRAN_DIR pointing to the base MODTRAN 6.0 directory.

2. Run the following code:
```
$ cd $(isofit path examples)/20171108_Pasadena
$ ./modtran.sh
```

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist. It will call MODTRAN to rebuild them, which will take a few minutes.

4. Look for output data in `$(isofit path examples)/20171108_Pasadena/output/`

# Known Incompatibilities

Ray may have compatibility issues with older machines with glibc < 2.14.

# Additional Installation Info for Developers

Be sure to read the [contributing](../../developers/contributing.md) page as additional installation steps must be performed.
