# -*- coding: utf-8 -*-
from email.policy import default
import click
import logging
import pandas as pd
import sys
import os
from pathlib import Path
from preprocess import preprocess_data, preprocess_target, extract_target
from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils import save_as_pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--output_data_filepath', type=click.Path(), default=True)
@click.option('--output_target_filepath', type=click.Path(), default=None)
def main(input_filepath, output_data_filepath=None, output_target_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    df = pd.read_csv(input_filepath)
    df = preprocess_data(df)
    if output_target_filepath:
        df, target = extract_target(df)
        target = preprocess_target(target)
        save_as_pickle(target, output_target_filepath)

    save_as_pickle(df, output_data_filepath)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
