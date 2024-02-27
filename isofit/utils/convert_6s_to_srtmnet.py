import click
import dask.array as da
import numpy as np
import xarray as xr


def predict(emulator, sixs, resample, scaler, output):
    """ """

    # Delay expensive import
    from tensorflow import keras

    data = sixs.to_array("quantity").stack(stack=["quantity", "wl"])

    # Now predict, scale, and add the interpolations
    print("Loading and predicting with emulator")
    emulator = keras.models.load_model(emulator)
    predicts = da.from_array(emulator.predict(data))
    predicts /= scaler
    predicts += resample

    # Unstack back to a dataset and save
    predicts = predicts.unstack("stack").to_dataset("quantity")

    print(f"Saving predicts to: {output}")
    predicts.to_netcdf(output)


@click.command(name="6S_to_sRTMnet")
@click.argument("sixs")  # , help="6S input LUT netCDF")
@click.argument("emulator")  # , help="Path to sRTMnet emulator")
@click.option("-a", "--aux", help="Emulator aux file")
@click.option(
    "-o",
    "--output",
    help="Output resampled 6S to sRTMnet",
    default="sRTMnet.resample.nc",
)
@click.option(
    "-p", "--predicts", help="Output sRTMnet predicts", default="sRTMnet.predicts.nc"
)
def cli_6s_to_srtmnet(sixs, emulator, aux, output, predicts):
    """\
    Converts 6S LUT outputs to sRTMnet input
    """
    print("Loading emulator auxiliary file")
    aux = np.load(aux)
    scaler = aux.get("response_scaler", 100.0)

    # Lazy load so it's not in memory
    print("Lazy loading 6S LUT")
    sixs = xr.open_mfdataset([sixs])

    # Select only the rt quantities of interest
    sixs = sixs[aux["rt_quantities"]]

    print("Interpolating from 6S wavelengths to emulator's")
    resample = sixs.interp({"wl": aux["emulator_wavelengths"]})

    # Perform Dask operations
    print(f"Saving resample to: {output}")
    resample.to_netcdf(output)

    # Stack the quantities together along a new dimension named `quantity`
    resample = resample.to_array("quantity").stack(stack=["quantity", "wl"])

    # Predict on the input data
    predict(emulator, sixs, resample, scaler, predicts)

    print("Done")
