#!/usr/bin/env python
# Utility Functions to install and run GPTs to 
# clean up transcriptions from the YouTube videos 
# associated with the Vervaeke Foundation.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

"""
TranscriptionUtilities: A set of utility functions 
common to installing and running GPTs, cleaning up 
transcriptions from YouTube videos, and performing 
speech-to-text on audio.
"""

from difflib import SequenceMatcher
from typing import List, Optional
try: import dill as pickle
except:
    try: import pickle5 as pickle
    except: import pickle
import pandas as pd
import os
import sys
import subprocess


import warnings
warnings.filterwarnings("ignore")

class TranscriptionUtilities(object):
    """
    This class implements the core of the utility
    functions needed to install and run GPTs to 
    clean up transcriptions from the YouTube 
    videos associated with the Vervaeke Foundation.
    
    Examples
    --------
    
    import sys
    import os
    sys.path.insert(1, os.path.abspath('../py'))
    from nt_utils import TranscriptionUtilities
    
    tu = TranscriptionUtilities(
        data_folder_path=os.path.abspath('../data'),
        saves_folder_path=os.path.abspath('../saves')
    )
    """
    
    def __init__(self, data_folder_path=None, saves_folder_path=None, verbose=False):
        self.verbose = verbose
        self.update_modules_list(verbose=verbose)
        
        # Create the data folder if it doesn't exist
        if data_folder_path is None:
            self.data_folder = '../data'
        else:
            self.data_folder = data_folder_path
        os.makedirs(self.data_folder, exist_ok=True)
        if verbose:
            print('data_folder: {}'.format(os.path.abspath(self.data_folder)))
        
        # Create the saves folder if it doesn't exist
        if saves_folder_path is None:
            self.saves_folder = '../saves'
        else:
            self.saves_folder = saves_folder_path
        os.makedirs(self.saves_folder, exist_ok=True)
        if verbose:
            print('saves_folder: {}'.format(os.path.abspath(self.saves_folder)))
        
        # Create the assumed directories
        
        self.data_models_folder = os.path.join(self.data_folder, 'models')
        os.makedirs(name=self.data_models_folder, exist_ok=True)
        
        self.saves_pickle_folder = os.path.join(self.saves_folder, 'pkl')
        os.makedirs(name=self.saves_pickle_folder, exist_ok=True)
        
        self.saves_text_folder = os.path.join(self.saves_folder, 'txt')
        os.makedirs(name=self.saves_text_folder, exist_ok=True)
        
        self.saves_mp3_folder = os.path.join(self.saves_folder, 'mp3')
        os.makedirs(name=self.saves_mp3_folder, exist_ok=True)
        
        self.saves_wav_folder = os.path.join(self.saves_folder, 'wav')
        os.makedirs(name=self.saves_wav_folder, exist_ok=True)
        
        self.anaconda_folder = os.path.dirname(sys.executable)
        self.scripts_folder = os.path.join(self.anaconda_folder, 'Scripts')
        if self.scripts_folder not in sys.path:
            sys.path.insert(1, self.scripts_folder)

        # Handy list of the different types of encodings
        self.encoding_type = ['latin1', 'iso8859-1', 'utf-8'][2]

    def save_transcript(self, video_id, file_name=None):
        self.ensure_module_installed('youtube-transcript-api', upgrade=True, verbose=False)
        from youtube_transcript_api import YouTubeTranscriptApi

        # Get the transcription for the video
        transcript_dicts_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_str = ' '.join([transcript_dict['text'] for transcript_dict in transcript_dicts_list]).lower().strip()

        # Get the filepath
        if file_name is None:
            file_name = video_id
        if not file_name.endswith('.txt'):
            file_name = f'{file_name}.txt'
        import re
        file_name = re.sub('[^A-Za-z0-9\.]+', '_', file_name)
        file_path = os.path.join(self.saves_text_folder, file_name)

        # Print transcript to file
        with open(file_path, 'w') as f:
            print(transcript_str, file=f)

        # Open in Notepad++
        text_editor_path = 'C:\\Program Files\\Notepad++\\notepad++.exe'
        command_str = f'"{text_editor_path}" "{file_path}"'
        output_str = subprocess.check_output(command_str.split(' '))
    
    def similar(self, a: str, b: str) -> float:
        """
        Compute the similarity between two strings.

        Parameters
        ----------
        a : str
            The first string.
        b : str
            The second string.

        Returns
        -------
        float
            The similarity between the two strings, as a float between 0 and 1.
        """

        return SequenceMatcher(None, str(a), str(b)).ratio()
    
    def get_row_dictionary(self, value_obj, row_dict={}, key_prefix=''):
        '''
        This function takes a value_obj (either a dictionary, list or scalar value) and creates a flattened dictionary from it, where
        keys are made up of the keys/indices of nested dictionaries and lists. The keys are constructed with a key_prefix
        (which is updated as the function traverses the value_obj) to ensure uniqueness. The flattened dictionary is stored in the
        row_dict argument, which is updated at each step of the function.

        Parameters
        ----------
        value_obj : dict, list, scalar value
            The object to be flattened into a dictionary.
        row_dict : dict, optional
            The dictionary to store the flattened object.
        key_prefix : str, optional
            The prefix for constructing the keys in the row_dict.

        Returns
        ----------
        row_dict : dict
            The flattened dictionary representation of the value_obj.
        '''
        
        # Check if the value is a dictionary
        if type(value_obj) == dict:
            
            # Iterate through the dictionary 
            for k, v, in value_obj.items():
                
                # Recursively call get_row_dictionary() with the dictionary key as part of the prefix
                row_dict = get_row_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}_{k}'
                )
                
        # Check if the value is a list
        elif type(value_obj) == list:
            
            # Get the minimum number of digits in the list length
            list_length = len(value_obj)
            digits_count = min(len(str(list_length)), 2)
            
            # Iterate through the list
            for i, v in enumerate(value_obj):
                
                # Add leading zeros to the index
                if (i == 0) and (list_length == 1):
                    i = ''
                else:
                    i = str(i).zfill(digits_count)
                
                # Recursively call get_row_dictionary() with the list index as part of the prefix
                row_dict = get_row_dictionary(
                    v, row_dict=row_dict, key_prefix=f'{key_prefix}{i}'
                )
                
        # If value is neither a dictionary nor a list
        else:
            
            # Add the value to the row dictionary
            if key_prefix.startswith('_') and (key_prefix[1:] not in row_dict):
                key_prefix = key_prefix[1:]
            row_dict[key_prefix] = value_obj
        
        return row_dict

    def pickle_exists(self, pickle_name: str) -> bool:
        """
        Checks if a pickle file exists.

        Parameters
        ----------
        pickle_name : str
            The name of the pickle file.

        Returns
        -------
        bool
            True if the pickle file exists, False otherwise.
        """

        pickle_path = os.path.join(self.saves_pickle_folder, '{}.pkl'.format(pickle_name))

        return os.path.isfile(pickle_path)

    def load_object(self, obj_name: str, pickle_path: str = None, download_url: str = None, verbose: bool = False) -> object:
        """
        Load an object from a pickle file.

        Parameters
        ----------
        obj_name : str
            The name of the object to load.
        pickle_path : str, optional
            The path to the pickle file. Defaults to None.
        download_url : str, optional
            The URL to download the pickle file from. Defaults to None.
        verbose : bool, optional
            Whether to print status messages. Defaults to False.

        Returns
        -------
        object
            The loaded object.
        """

        if pickle_path is None:
            pickle_path = os.path.join(self.saves_pickle_folder, '{}.pkl'.format(obj_name))

        if not os.path.isfile(pickle_path):
            if isinstance(object, pd.DataFrame):
                self.attempt_to_pickle(object, pickle_path, raise_exception=False)
            else:
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)

        else:
            try:
                object = pd.read_pickle(pickle_path)
            except:
                with open(pickle_path, 'rb') as handle:
                    object = pickle.load(handle)

        if verbose:
            print('Loaded object {} from {}'.format(obj_name, pickle_path))

        return(object)
    
    def store_objects(self, verbose: bool = True, **kwargs: dict) -> None:
        """
        Store objects to pickle files.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print status messages. Defaults to True.
        **kwargs : dict
            The objects to store. The keys of the dictionary are the names of the objects, and the values are the objects themselves.

        Returns
        -------
        None

        """

        for obj_name in kwargs:
            # if hasattr(kwargs[obj_name], '__call__'):
            #     raise RuntimeError('Functions cannot be pickled.')
            pickle_path = os.path.join(self.saves_pickle_folder, '{}.pkl'.format(obj_name))
            if isinstance(kwargs[obj_name], pd.DataFrame):
                self.attempt_to_pickle(kwargs[obj_name], pickle_path, raise_exception=False, verbose=verbose)
            else:
                if verbose:
                    print('Pickling to {}'.format(os.path.abspath(pickle_path)))
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(kwargs[obj_name], handle, min(4, pickle.HIGHEST_PROTOCOL))

    def attempt_to_pickle(self, df: pd.DataFrame, pickle_path: str, raise_exception: bool = False, verbose: bool = True) -> None:
        """
        Attempts to pickle a DataFrame to a file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to pickle.
        pickle_path : str
            The path to the pickle file.
        raise_exception : bool, optional
            Whether to raise an exception if the pickle fails. Defaults to False.
        verbose : bool, optional
            Whether to print status messages. Defaults to True.

        Returns
        -------
        None

        """

        try:
            if verbose:
                print('Pickling to {}'.format(os.path.abspath(pickle_path)))
            df.to_pickle(pickle_path, protocol=min(4, pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            os.remove(pickle_path)
            if verbose:
                print(e, ": Couldn't save {:,} cells as a pickle.".format(df.shape[0]*df.shape[1]))
            if raise_exception:
                raise
    
    def update_modules_list(self, modules_list: Optional[List[str]] = None, verbose: bool = False) -> None:
        """
        Updates the list of modules that are installed.

        Parameters
        ----------
        modules_list : Optional[List[str]], optional
            The list of modules to update. If None, the list of installed modules will be used. Defaults to None.
        verbose : bool, optional
            Whether to print status messages. Defaults to False.

        Returns
        -------
        None
        """

        if modules_list is None:
            self.modules_list = [o.decode().split(' ')[0] for o in subprocess.check_output(f'{sys.executable} -m pip list'.split(' ')).splitlines()[2:]]
        else:
            self.modules_list = modules_list

        if verbose:
            print('Updated modules list to {}'.format(self.modules_list))
    
    def ensure_module_installed(self, module_name: str, upgrade: bool = False, verbose: bool = True) -> None:
        """
        Checks if a module is installed and installs it if it is not.

        Parameters
        ----------
        module_name : str
            The name of the module to check for.
        upgrade : bool, optional
            Whether to upgrade the module if it is already installed. Defaults to False.
        verbose : bool, optional
            Whether to print status messages. Defaults to True.

        Returns
        -------
        None
        """

        if module_name not in self.modules_list:
            command_str = f'{sys.executable} -m pip install {module_name}'
            if upgrade:
                command_str += ' --upgrade'
            if verbose:
                print(command_str)
            else:
                command_str += ' --quiet'
            output_str = subprocess.check_output(command_str.split(' '))
            if verbose:
                for line_str in output_str.splitlines():
                    print(line_str.decode())
            self.update_modules_list(verbose=verbose)