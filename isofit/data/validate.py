import click


@click.group("validate", invoke_without_command=True, no_args_is_help=True)
def cli():
    """\
    Validate extra ISOFIT files that do not come with the default installation
    """
    pass
