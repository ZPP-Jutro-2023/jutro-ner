from pathlib import Path
from spacy.cli import init_config, fill_config
from thinc.api import Config
import typer


def main(
    base_params_path: Path = typer.Argument(..., help="Path to base config parameters"),
    non_default_params: Path = typer.Argument(..., help="Path to extended config parameters"),
    output: Path = typer.Argument(..., help="Path to save resulting config")
) -> None:
    base_params = Config().from_disk(base_params_path)['base_params']

    base_config = init_config(
        lang=base_params['lang'],
        pipeline=base_params['pipeline'],
        optimize=base_params['optimize'],
        gpu=base_params['gpu'],
        pretraining=base_params['pretraining']
    )

    extended_config = Config().from_disk(path=non_default_params, interpolate=False)
    final_config = Config(base_config).merge(extended_config)
    final_config.to_disk(output)

    # Note: This is a workaround solution, would be better to pass Config object to validating
    # function instead of creating a file (required by spacy's fill_config).
    # Make sure that the final config is complete. If not, fill it with default values
    fill_config(output, output, pretraining=base_params['pretraining'])


if __name__ == '__main__':
    typer.run(main)
