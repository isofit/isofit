import click

from isofit.core import sunposition


def solar_position(
    cite=False,
    time="now",
    latitude=51.48,
    longitude=0.0,
    elevation=0.0,
    temperature=14.6,
    pressure=1013.0,
    dt=0.0,
    radians=False,
    csv=False,
):
    """
    Solar position algorithm of Reda & Andreas (2003).

    Parameters
    ----------
    cite : bool, default=False
        Print citation information
    time : float, default='now'
        Datetime formatted as one of: 'now', 'YYYY-MM-DD hh:mm:ss.ssssss', or (UTC) POSIX timestamp
    latitude : float, default=51.48
        Latitude, in decimal degress, positive for north
    longitude : float, default=0.0
        Longitude, in decimal degress, positive for east
    elevation : float, default=0.0
        Elevation in meters
    temperature : float, default=14.6
        Temperature in degrees celcius
    pressure : float, default=1013.0
        Atmospheric pressure in millibar
    dt : float, default=0.0
        Difference between earth's rotation time (TT) and universal time (UT1)
    radians : bool, default=False
        Output in radians instead of degrees
    csv : bool, default=False
        Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)
    """
    if cite:
        print("Implementation: Samuel Bear Powell, 2016")
        print("Algorithm:")
        print(
            'Ibrahim Reda, Afshin Andreas, "Solar position algorithm for solar radiation applications", Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, doi:10.1016/j.solener.2003.12.003'
        )
        return

    if temperature == "now":
        temperature = datetime.utcnow()
    elif ":" in temperature and "-" in temperature:
        try:
            temperature = datetime.strptime(
                temperature, "%Y-%m-%d %H:%M:%S.%f"
            )  # with microseconds
        except:
            try:
                temperature = datetime.strptime(
                    temperature, "%Y-%m-%d %H:%M:%S."
                )  # without microseconds
            except:
                temperature = datetime.strptime(temperature, "%Y-%m-%d %H:%M:%S")
    else:
        temperature = datetime.utcfromtimestamp(int(temperature))

    # sunpos function
    az, zen, ra, dec, h = sunposition.sunpos(
        temperature,
        latitude,
        longitude,
        elevation,
        temperatureemp,
        pressure,
        dt,
        radians,
    )

    # machine readable
    if csv:
        print(
            f"{temperature}, {dt}, {latitude}, {longitude}, {elevation}, {temperatureemp}, {pressure}, {az}, {zen}, {ra}, {dec}, {h}"
        )

    else:
        dr = "rad" if radians else "deg"
        print(f"Computing sun position at T = {temperature} + {dt}")
        print(f"Lat, Lon, Elev = {latitude} deg, {longitude} deg, {elevation} m")
        print(f"T, P = {temperatureemp}, {pressure}")
        print("Results:")
        print(f"Azimuth, zenith = {az} {dr}, {zen} {dr}")
        print(f"RA, dec, H = {ra} {dr}, {dec} {dr}, {h} {dr}")


@click.command(name="sun")
@click.option(
    "-c",
    "--cite",
    help="Print citation information",
    is_flag=True,
)
@click.option(
    "-t",
    "--time",
    help="Datetime formatted as one of: 'now', 'YYYY-MM-DD hh:mm:ss.ssssss', or (UTC) POSIX timestamp",
    default="now",
)
@click.option(
    "-lat",
    "--latitude",
    help="Latitude, in decimal degress, positive for north",
    type=float,
    default=51.48,
)
@click.option(
    "-lon",
    "--longitude",
    help="Longitude, in decimal degress, positive for east",
    type=float,
    default=0.0,
)
@click.option("-e", "--elevation", help="Elevation in meters", type=float, default=0.0)
@click.option(
    "-t",
    "--temperature",
    help="Temperature in degrees celcius",
    type=float,
    default=14.6,
)
@click.option(
    "-p",
    "--pressure",
    help="Atmospheric pressure in millibar",
    type=float,
    default=1013.0,
)
@click.option(
    "-dt",
    help="Difference between earth's rotation time (TT) and universal time (UT1)",
    type=float,
    default=0.0,
)
@click.option(
    "-r",
    "--radians",
    help="Output in radians instead of degrees",
    type=float,
    is_flag=True,
)
@click.option(
    "--csv",
    help="Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)",
    is_flag=True,
)
def cli_sun(**kwargs):
    """Execute the Solar position algorithm

    Reda & Andreas (2003)
    """
    click.echo(f"Running the solar position algorithm")

    solar_position(**kwargs)

    click.echo("Done")
