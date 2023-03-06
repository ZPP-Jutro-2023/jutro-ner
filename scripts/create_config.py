from spacy.cli import init_config, fill_config
from thinc.api import Config
from pathlib import Path
import typer

def main(
    base_params: Path           = typer.Argument(..., help="Path to base config parameters"),
    non_default_params: Path    = typer.Argument(..., help="Path to extended config parameters"),
    output: Path                = typer.Argument(..., help="Path to save resulting config")
) -> None:
    base_params = Config().from_disk(base_params)['base_params']

    base_config = init_config(
        lang        = base_params['lang'],
        pipeline    = base_params['pipeline'],
        optimize    = base_params['optimize'],
        gpu         = base_params['gpu'],
        pretraining = base_params['pretraining']
    )

    extended_config = Config().from_disk(path=non_default_params, interpolate=False)
    final_config = Config(base_config).merge(extended_config)
    final_config.to_disk(output)

    # Make sure that the final config is complete. If not, fill it with default values
    fill_config(output, output, pretraining=base_params['pretraining])

if __name__ == '__main__':
    typer.run(main)

