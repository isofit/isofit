# Building Examples

ISOFIT examples rely on the `isofit build` command to generate configuration files and scripts dependent on a user's active INI file. Each example contains a set of template files generate the required files for the example. By default, a user will not need to modify these templates. If an advanced user desires to change the configuration of an example, it is strongly recommended to run the build command first and edit the generated outputs. However, every example should work out-of-the-box with the default downloads and build.

## Developers

This section is specifically for developers seeking to expand either the examples.

### Creating Examples

ISOFIT leverages specially-designed templates to build the example configurations depending on the installation environment defined by an INI.
Creating a new example must define one or more templates for the given example type.

### Templates

Templates are used to generate configuration and script files relative to a user's installation environment. Changes to the ISOFIT INI may rebuild the examples quickly for a new environent. Instead of hardcoding relative paths, the `isofit build` command will replace values within the templates with the values defined by a given INI. For example, a template may define `{examples}`, this will be replaced with the INI's `examples` string.

There are two types of examples supported at this time:

1. Direct `Isofit` calls. These examples build configuration files to pass directly into the `Isofit` class to call `.run()`

For existing examples of this type include [SantaMonica](https://github.com/isofit/isofit-tutorials/tree/main/20151026_SantaMonica), [Pasadena](https://github.com/isofit/isofit-tutorials/tree/main/20171108_Pasadena), and [ThermalIR](https://github.com/isofit/isofit-tutorials/tree/main/20190806_ThermalIR). Depending on the example, extra directories may be included such as prebuilt simulation files in the `lut` directory.

A bash and python script will be generated for each directory under the templates directory. For example, given a template directory:

```
[example]/
└─ templates/
  ├─ reduced/
  | ├─ config1.json
  | └─ config2.json
  ├─ advanced/
  | └─ config3.yml
  └─ surface.json
```

will generate the following configs and scripts:

```
[example]/
├─ configs/
| ├─ reduced/
| | ├─ config1.json
| | └─ config2.json
| ├─ advanced/
| | └─ config3.json
| └─ surface.json
├─ reduced.sh
├─ reduced.py
├─ advanced.sh
└─ advanced.py
```

Each script will have the configs for it. For example, `reduced.sh` would contain:

```bash
# Build a surface model first
echo 'Building surface model: surface.json'
isofit surface_model ~/.isofit/examples/[example]/configs/surface.json

# Now run retrievals
echo 'Running 1/2: config1.json'
isofit run --level DEBUG ~/.isofit/examples/[example]/configs/reduced/config1.json

echo 'Running 2/2: config2.json'
isofit run --level DEBUG ~/.isofit/examples/[example]/configs/reduced/config2.json
```

2. `apply_oe` scripts. These examples use templates to define the arguments for a call to the `isofit apply_oe` utility.

Existing examples of this type include the [small](https://github.com/isofit/isofit-tutorials/tree/main/image_cube/small/templates) and [medium image cube](https://github.com/isofit/isofit-tutorials/tree/main/image_cube/medium/templates) examples. These templates are a list of arguments in a `[name].args.json` file. For each `[name]` file, separate scripts will be generated. For example, given the following templates:

```
[example]/
└─ templates/
  ├─ simple.args.json
  └─ advanced.args.json
```

will generate the following scripts:

```
[example]/
├─ simple.sh
└─ advanced.sh
```
The small image cube example's `default.args.json` is currently defined as:

```json
[
"{imagecube}/medium/ang20170323t202244_rdn_7k-8k",
"{imagecube}/medium/ang20170323t202244_loc_7k-8k",
"{imagecube}/medium/ang20170323t202244_obs_7k-8k",
"{examples}/image_cube/medium",
"ang",
"--surface_path {examples}/image_cube/medium/configs/surface.json",
"--emulator_base {srtmnet}/sRTMnet_v120.h5",
"--n_cores {cores}",
"--presolve",
"--segmentation_size 400",
"--pressure_elevation"
]
```

This will generate `default.sh`:

```
isofit apply_oe \
  ~/.isofit/examples/imagecube/small/ang20170323t202244_rdn_7000-7010 \
  ~/.isofit/examples/imagecube/small/ang20170323t202244_loc_7000-7010 \
  ~/.isofit/examples/imagecube/small/ang20170323t202244_obs_7000-7010 \
  ~/.isofit/examples/examples/image_cube/small \
  ang \
  --surface_path ~/.isofit/examples/examples/image_cube/small/configs/surface.json \
  --n_cores 10 \
  --presolve \
  --segmentation_size 400 \
  --pressure_elevation
```

Once the the example with its templates are finalized, it must be integrated into the [ISOFIT Tutorials](https://github.com/isofit/isofit-tutorials) repository. Create a new pull request with a description of the example being created and maintainers will review it then merge and release a new version.
