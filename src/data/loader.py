import json
import re
import ast
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

import requests
import time
import world_bank_data as wbd
import yfinance as yf
from urllib.parse import quote

import os
import dotenv
dotenv.load_dotenv()

from .log_utilities import _find_project_root, setup_logging, APICallLogger
from .data_utilities import upload_to_huggingface

username = os.getenv("HUGGINGFACE_USERNAME")
if not username:
    raise ValueError("HUGGINGFACE_USERNAME not set in .env")

root_dir = _find_project_root()

class TopicClassifier:
    """
    Classifier for categorizing variables into predefined buckets.
    
    This classifier categorizes variables based on Stock & Watson (1989) 
    methodology, using their names and attributes. It uses a configuration 
    file to define buckets, keywords, and topic overrides for different data sources.
    
    Attributes:
    -----------
    config (Dict[str, Any])
        Configuration loaded from JSON file.
    buckets (Dict[int, str])
        Mapping of bucket IDs to names.
    keywords (Dict[int, List[str]])
        Mapping of bucket IDs to lists of keywords (regex patterns).
    exclude_keywords (set)
        Set of keywords to exclude from classification.
    data_tag (str)
        Identifier for the data source (e.g., 'WORLDBANK').
    column_mappings (Dict[str, str])
        Mappings of standard column names to actual names in the dataset.
        Specific to data_tag.
    topic_overrides (Dict[str, int])
        Overrides for topics to specific buckets.
        Specific to data_tag.
    exclude_topics (set)
        Set of topics to exclude from classification.
        Specific to data_tag.
    logger (logging.Logger)
        Logger instance for logging messages.
    """
    
    def __init__(self, data_tag: str, config_path: str = "config.json"):
        """
        Initialize classifier with configuration from file.
        
        Parameters:
        -----------
        data_tag: str
            Identifier for the data source (e.g., 'WORLDBANK').
        config_path: str
            Path to the JSON configuration file.

        Raises:
        -------
        ValueError: If data_tag is not found in configuration.
        FileNotFoundError: If configuration file is not found.
        """

        # Load config file
        self.config = self._load_config(config_path)

        # Get variable classification buckets & keywords
        self.buckets = {int(k): v for k, v in self.config[
            "VARIABLE_CLASSIFICATION"]["BUCKETS"].items()
            }
        self.keywords = {int(k): v for k, v in self.config[
                "VARIABLE_CLASSIFICATION"]["KEYWORDS"].items()
                }
        self.exclude_keywords = set(
            self.config["VARIABLE_CLASSIFICATION"]["EXCLUDE_KEYWORDS"])
        
        # Set up data-specific configurations
        if data_tag.upper() not in self.config["DATA_TAGS"]:
            raise ValueError(
                f"Data tag '{data_tag}' not found in configuration"
            )
        else:
            self.data_tag = data_tag
            self.column_mappings = self.config[
                f"{self.data_tag.upper()}_COLUMN_MAPPINGS"]
            
            # Set topic overrides and exclude topics if present
            vc_config = self.config["VARIABLE_CLASSIFICATION"]
            topic_overrides_key = f"{self.data_tag.upper()}_TOPIC_OVERRIDES"
            exclude_topics_key = f"{self.data_tag.upper()}_EXCLUDE_TOPICS"

            self.topic_overrides = vc_config.get(topic_overrides_key, {})
            self.exclude_topics = set(vc_config.get(exclude_topics_key, []))

        # Setup logging
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.

        Returns:
        --------
        Dict
            Containing the loaded configuration.
            
        Raises:
        -------
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the JSON file contains invalid syntax.
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def set_column_mappings(self, mappings: Dict[str, str]) -> None:
        """
        Update column mappings for different data sources.
        
        Parameters:
        -----------
        mappings: Dict
            mapping standard names to actual column names.
            e.g., {"name_column": "indicator_name", "topics_column": "categories"}
        """
        self.column_mappings.update(mappings)
        self.logger.info(f"Updated column mappings: {mappings}")
    
    def classify_variable(self, row: pd.Series) -> int:
        """
        Classify a variable based on its attributes.

        Parameters:
        -----------
        row: pd.Series
            Containing variable information.
            
        Returns:
        --------
        integer
            Bucket ID for the classification.
        """
        try:
            # Extract text fields using column mappings
            name = str(row.get(self.column_mappings["name_column"], "")).lower()
            note = str(row.get(self.column_mappings["note_column"], "")).lower()
            topics_raw = str(row.get(self.column_mappings["topics_column"], ""))
            
            # Parse topics
            topics = set(t.strip() for t in topics_raw.split(',') if t.strip())
            
            # Exclude irrelevant topics
            if topics & self.exclude_topics:
                return -1
            
            # Combine text for keyword matching
            text = f"{name} {note}"
            
            # Keyword-based elimination
            for pattern in self.exclude_keywords:
                if re.search(pattern, text, re.IGNORECASE):
                    return -1

            # Keyword-based classification
            for bucket_id, patterns in self.keywords.items():
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    return bucket_id
                
            # Apply topic-based overrides
            for topic, bucket in self.topic_overrides.items():
                if topic in topics:
                    return bucket
            
            # Default as other
            return -1
            
        except Exception as e:
            self.logger.error(f"Error classifying variable: {e}")
            return -1
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all variables in a dataframe.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Containing variables to classify.

        Returns:
        --------
        pd.DataFrame
            With added 'bucket' and 'bucket_name' columns.
        """
        df = df.copy()
        
        # Apply classification
        df['bucket'] = df.apply(self.classify_variable, axis=1)
        df['bucket_name'] = df['bucket'].map(self.buckets)
        
        # Log classification summary
        classification_counts = df['bucket_name'].value_counts()
        self.logger.info(f"Classification summary:\n{classification_counts}")
        
        return df
    
    def add_classification_rules(
            self, bucket_id: int, keywords: List[str], 
            topic_overrides: Optional[Dict[str, int]] = None) -> None:
        """
        Add new classification rules.

        Parameters:
        -----------
        bucket_id: int
            Bucket number for classification.
        keywords: List[str]
            Of regex patterns for keyword matching.
        topic_overrides: Optional[Dict[str, int]]
            Topic-to-bucket mappings.
        """
        self.keywords[bucket_id] = keywords
        if topic_overrides:
            self.topic_overrides.update(topic_overrides)
        self.logger.info(f"Added classification rules for bucket {bucket_id}")


class DataLoader(ABC):
    """
    Abstract base class for data loaders with common functionality.
    
    This class provides a template for loading data from various sources,
    classifying variables, and uploading to HuggingFace Hub.
    
    Attributes:
    -----------
    logger (logging.Logger)
        Logger instance for logging messages.
    data_tag (str)
        Identifier for the data source (e.g., 'WORLDBANK').
    config (Dict[str, Any])
        Configuration loaded from JSON file.
    classifier (TopicClassifier)
        Instance of TopicClassifier for classifying variables.
    """
    
    def __init__(self, data_tag: str, config_path: str = "config.json"):
        """
        Initialize DataLoader with data tag and configuration.
        
        Parameters:
        -----------
        data_tag: str
            Identifier for the data source (e.g., 'WORLDBANK').
        config_path: str
            Path to the JSON configuration file.
            
        Raises:
        -------
        ValueError: If data_tag is not found in configuration.
        """
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")
        self.data_tag = data_tag.upper()
        self.config = self._load_config(config_path)
        self.developed_markets = self.config["DEVELOPED_MARKETS"]
        self.emerging_markets = self.config["EMERGING_MARKETS"]
        self.classifier = TopicClassifier(data_tag, config_path)

        # Validate data tag exists in configuration
        if self.data_tag not in self.config["DATA_TAGS"]:
            raise ValueError(
                f"Data tag '{data_tag}' not found in configuration")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.
            
        Returns:
        --------
        Dict
            Containing the loaded configuration.
            
        Raises:
        -------
        Exception: If configuration loading fails.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _get_file_prefix(self) -> str:
        """
        Get standardized file prefix based on data tag.
        
        Returns:
        --------
        str
            Lowercase version of the data tag for consistent file naming.
        """
        return self.data_tag.lower()
    
    def _get_metadata_file(self, variables_file: str = None) -> str:
        """
        Get the variable (metadata) file path with standardized prefix.
        
        Parameters:
        -----------
        variables_file: str
            Optional path to existing variables file.
            
        Returns:
        --------
        str
            Path to the metadata file.
            
        Raises:
        -------
        FileNotFoundError: If metadata file is not found.
        """
        if variables_file is None or not os.path.exists(variables_file):
            variables_file = f'{root_dir}/data/{self._get_file_prefix()}_variables.csv'
            if not os.path.exists(variables_file):
                self.logger.error(
                    f"Metadata file '{variables_file}' not found.")
                raise FileNotFoundError(
                    f"Metadata file '{variables_file}' not found.")
        return variables_file
    
    @abstractmethod
    def fetch_indicators(self) -> pd.DataFrame:
        """
        Fetch indicators from data source.
        
        Must be implemented by subclasses.
        
        Returns:
        --------
        pd.DataFrame
            Containing indicator metadata.
        """
        pass
    
    @abstractmethod
    def download_series_data(self,
                             variables: pd.DataFrame = None) -> pd.DataFrame:
        """
        Download time series data.
        
        Must be implemented by subclasses.

        Parameters:
        -----------
        variables: pd.DataFrame
            Containing variable information.

        Returns:
        --------
        pd.DataFrame
            Containing time series data.
        """
        pass
    
    def classify_and_save_variables(self,
                                    predictors: pd.DataFrame,
                                    save_metadata: bool = True) -> pd.DataFrame:
        """
        Common method for classifying variables.
        
        Parameters:
        -----------
        predictors: pd.DataFrame
            Containing predictor variables to classify.
        save_metadata: bool
            Whether to save classified variables to CSV.

        Returns:
        --------
        pd.DataFrame
            With classified variables (excluding 'Others' category).

        Raises:
        -------
        Exception: If saving classified variables fails.
        """
        output_file = f'{root_dir}/data/{self._get_file_prefix()}_variables.csv'
        
        # Classify variables using inherited classifier
        classified_predictors = self.classifier.classify_dataframe(predictors)
        
        # Save to CSV
        if save_metadata:
            try:
                classified_predictors.to_csv(output_file, index=False)
                self.logger.info(
                    f"Classified variables saved to {output_file}")
            except Exception as e:
                self.logger.error(
                f"Failed to save classified variables: {e}")
                raise e

        # Remove "Others" category
        initial_count = len(classified_predictors)
        classified_predictors = classified_predictors[
            classified_predictors['bucket_name'] != "Others"
        ]
        
        self.logger.info(
            f"Removed {initial_count - len(classified_predictors)} 'Others' variables")
        
        return classified_predictors
    
    def run_data_pipeline(self, save_metadata=True, upload=True):
        """
        Execute the complete data download and processing pipeline.
        
        This method orchestrates the entire workflow:
        1. Fetch indicators from API.
        2. Classify variables using the TopicClassifier.
        3. Download time series data for classified variables.
        4. Upload the final dataset to Hugging Face Hub.
        
        Parameters:
        -----------
        save_metadata: bool
            Whether to save intermediate metadata files.
        upload: bool
            Whether to upload the final dataset to Hugging Face Hub.

        Returns:
        --------
        Tuple
            containing the final processed datasets (developed_data, emerging_data).
            
        Raises:
        -------
        ValueError: If required environment variables are not set.
        Exception: If any step in the pipeline fails.
        """
        self.logger.info(f"Starting {self.data_tag} download pipeline.")

        try:
            # Step 1: Fetch indicators
            indicators = self.fetch_indicators()
            if save_metadata:
                indicators.to_csv(
                    f'{root_dir}/data/{self._get_file_prefix()}_variables.csv',
                    index=False)
            
            # Step 2: Classify variables
            predictors = self.classify_and_save_variables(indicators, save_metadata)
            
            # Step 3: Download series data
            developed_data = self.download_series_data(predictors, self.developed_markets)
            emerging_data = self.download_series_data(predictors, self.emerging_markets)
            
            # Step 4: Upload to Hugging Face (uses base class method)
            if upload:
                repo_name = f"{username}/{self._get_file_prefix()}-download"
                data = developed_data.set_index('Date')
                upload_to_huggingface(data, repo_name, config_name="developed")
                data = emerging_data.set_index('Date')
                upload_to_huggingface(data, repo_name, config_name="emerging")

            self.logger.info(
                f"{self.data_tag} download pipeline completed. Final dataset shape: Developed = {developed_data.shape}, Emerging = {emerging_data.shape}")
            
            return developed_data, emerging_data

        except Exception as e:
            self.logger.error(f"{self.data_tag} download pipeline failed: {e}")
            raise


class WorldBankDataLoader(DataLoader):
    """
    World Bank specific data loader that inherits from DataLoader.
    
    This class implements methods to fetch indicators, download series
    data, classify variables, and run the full data processing pipeline.

    Attributes:
    -----------
    config_path (str)
        Path to the configuration file.
    topics (List[str])
        List of topics to filter indicators.
    sources (List[str])
        List of data sources to fetch indicators.
    developed_markets (List[str])
        List of developed countries for data filtering.
    emerging_markets (List[str])
        List of emerging markets for data filtering.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize WorldBankDataLoader with configuration.
        
        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.
        """
        # Call parent constructor with WORLDBANK data tag
        super().__init__("WORLDBANK", config_path)
        
        # Extract World Bank specific configuration
        self.topics = self.config["WORLDBANK_TOPICS"]
        self.sources = self.config["WORLDBANK_SOURCES"]

        self.logger.info("WorldBankDataLoader initialized successfully")
    
    def fetch_indicators(self) -> pd.DataFrame:
        """
        Fetch indicators from World Bank API with logging.
        
        This method retrieves indicators from specified sources, 
        filters them by topics, and removes duplicates. It uses the 
        APICallLogger for detailed logging of API calls.
        
        Returns:
        --------
        pd.DataFrame
            Containing fetched and filtered indicators.
        """
        predictors = pd.DataFrame()
        
        # Get source IDs
        source_ids = []
        for source in self.sources:
            with APICallLogger(
                self.logger, f"get_sources for {source}", source=source):
                try:
                    sources_df = wbd.get_sources(source=source)
                    source_id = sources_df.index[sources_df['name'] == source][0]
                    source_ids.append(source_id)
                    self.logger.info(f"Found source ID {source_id} for {source}")
                except Exception as e:
                    self.logger.error(f"Failed to get source ID for {source}: {e}")
                    continue
        
        # Fetch indicators for each source
        for source, source_id in zip(self.sources, source_ids):
            with APICallLogger(
                self.logger, f"get_indicators",
                source=source, source_id=source_id):
                try:
                    indicators = wbd.get_indicators(
                        source=source_id).reset_index()
                    
                    # Filter by topics if available
                    if 'topics' in indicators.columns and any(
                        indicators['topics'] != ""):
                        df = indicators[indicators['topics'].apply(
                            lambda x: any(str(topic) in str(x) for topic in self.topics) 
                            if pd.notnull(x) else False
                        )]
                        self.logger.info(
                            f"Filtered {len(df)} indicators from {source} by topics")
                    else:
                        df = indicators
                        self.logger.info(
                            f"Using all {len(df)} indicators from {source}")
                    
                    df = df.copy()
                    df['source_id'] = source_id
                    predictors = pd.concat(
                        [predictors, df], ignore_index=True)
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to fetch indicators from {source}: {e}")
                    continue
        
        # Remove duplicates
        initial_count = len(predictors)
        predictors.drop_duplicates(keep='first', inplace=True)
        predictors.reset_index(drop=True, inplace=True)
        
        self.logger.info(
            f"Removed {initial_count - len(predictors)} duplicates, "
            f"final count: {len(predictors)} indicators")
        
        return predictors

    def download_series_data(self,
                             variables: pd.DataFrame = None,
                             countries: List[str] = None) -> pd.DataFrame:
        """
        Download time series data for classified variables.
        
        This method reads a CSV file containing variable IDs,
        fetches the time series data for each variable from 
        the World Bank API, and merges them into a master 
        DataFrame. It uses the APICallLogger for detailed 
        logging of API calls and handles errors.
        
        Parameters:
        -----------
        variables: pd.DataFrame
            Containing variable information. If None, loads from metadata file.
                
        Returns:
        --------
        pd.DataFrame
            Containing downloaded time series data.
        """
        if variables is None or variables.empty:
            variables_file = self._get_metadata_file()
            variables = pd.read_csv(variables_file)
            self.logger.info(
                f"Loaded {len(variables)} variables from {variables_file}")
        
        if countries is None or not countries:
            countries = self.developed_markets + self.emerging_markets
            self.logger.info(
                f"No countries specified, using all developed and emerging markets.")
        
        master_data = pd.DataFrame()
        successful_downloads = 0
        
        all_data = []

        for _, row in variables.iterrows():
            
            # First attempt: Use primary method
            try:
                data = wbd.get_series(
                    indicator=row['id'],
                    country=countries,
                    simplify_index=False
                )\
                .reset_index()\
                .drop(columns=['Series'])

                successful_downloads += 1
                
            except Exception as e:
                # Second attempt: Use fallback method
                try:
                    data = self.get_series(
                        source_id=row['source_id'],
                        indicator_id=row['id'],
                        country=countries
                    ).drop(columns=['Series'])

                    successful_downloads += 1

                except Exception as e:
                    continue

            # Rename the value column to the indicator
            data = data.rename(columns={data.columns[-1]: row['id']})
            all_data.append(data)

        # Merge all DataFrames on Country and Year
        if all_data:
            master_data = all_data[0]
            for df in all_data[1:]:
                master_data = master_data.merge(df, on=['Country', 'Year'], how='outer')
        master_data.rename(columns={'Year': 'Date'}, inplace=True)

        # Log final statistics
        self.logger.info(
            f"Data download summary: "
            f"{len(variables)} attempted, "
            f"{successful_downloads} successful, "
            f"{len(variables) - successful_downloads} failed")
        
        return master_data
    
    def get_series(self, source_id, indicator_id: str,
                   country: List[str] = None,
                   start_year: int = 1900, end_year: int = 2025) -> pd.DataFrame:
        """
        Alternative method to fetch World Bank data using direct API calls.
        This method is used as a fallback when wbd.get_series() fails.
        
        Parameters:
        -----------
        source_id: int/str
            Source ID for the World Bank data
        indicator_id: str
            Indicator ID for the World Bank data
        country: List[str]
            List of countries to filter data
        start_year: int
            Start year for data retrieval
        end_year: int
            End year for data retrieval

        Returns:
        --------
        pd.DataFrame
            Processed data from World Bank API
        """
        
        base_url = "https://api.worldbank.org/v2"
        if country is None or not country:
            countries = ";".join(self.developed_markets + self.emerging_markets)
        else:
            countries = ";".join(country)
        
        url = f"{base_url}/country/{countries}/indicator/{indicator_id}"
        params = {
            'format': 'json',
            'date': f"{start_year}:{end_year}",
            'per_page': 10000,
            'source': str(source_id)
        }
        
        all_data = []
        page = 1
        
        while True:
            params['page'] = page
            
            try:
                self.logger.debug(f"Fetching page {page} from World Bank API")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if len(data) > 1 and data[1]:
                        all_data.extend(data[1])
                        self.logger.debug(
                            f"Retrieved {len(data[1])} records from page {page}")
                        
                        # Check for more pages
                        total_pages = data[0]['pages']
                        total_records = data[0]['total']
                        
                        self.logger.info(
                            f"Page {page}/{total_pages}, Total records: {total_records}")
                        
                        if page >= total_pages:
                            break
                        page += 1
                    else:
                        break
                        
                elif response.status_code == 429:
                    # Rate limit exceeded
                    self.logger.warning("Rate limit exceeded, waiting 5 seconds...")
                    time.sleep(5)
                    continue
                    
                else:
                    self.logger.error(
                        f"Failed to fetch data from API: {response.status_code} - {response.text}")
                    break
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Request timeout on page {page}")
                break
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed on page {page}: {e}")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error on page {page}: {e}")
                break
            
            # Rate limiting
            time.sleep(0.1)
        
        self.logger.info(f"Retrieved {len(all_data)} total records using fallback method")

        processed_records = []
        
        for record in all_data:
            processed_record = {
                'Country': record['country']['value'],
                'Year': record['date'],
                'Series': record['indicator']['id'],
                record['indicator']['id']: float(record['value']) if record[
                    'value'] not in [None, ''] else np.nan
            }
            processed_records.append(processed_record)
        df = pd.DataFrame(processed_records).reset_index(drop=True)

        self.logger.info(f"Processed {len(df)} valid records for {indicator_id}")
        
        return df


class OECDDataLoader(DataLoader):
    """
    OECD specific data loader that inherits from DataLoader.
    
    This class implements methods to fetch indicators, download series
    data, classify variables, and run the full data processing pipeline.

    Attributes:
    -----------
    config_path (str)
        Path to the configuration file.
    topics (List[str])
        List of topics to filter indicators.
    sources (List[str])
        List of data sources to fetch indicators.
    developed_markets (List[str])
        List of developed countries for data filtering.
    emerging_markets (List[str])
        List of emerging markets for data filtering.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize OECDDataLoader with configuration.
        
        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.
        """
        # Call parent constructor with OECD data tag
        super().__init__("OECD", config_path)

        # Extract OECD specific configuration
        self.topics = None
        self.sources = None

        self.logger.info("OECDDataLoader initialized successfully")

    def fetch_indicators(self, countries: List[str] = None) -> pd.DataFrame:
        """
        Fetch indicators from OECD API.
        
        This method retrieves indicators from specified sources,
        filters them by topics, and removes duplicates.
        
        Returns:
        --------
        pd.DataFrame
            Containing fetched and filtered indicators.
        """
        # Build the API request to get metadata for all series.
        if countries is None or not countries:
            countries = self.developed_markets + self.emerging_markets
        
        geo_filter = json.dumps(
            {"LOCATION": countries})
        meta_url = f"https://api.db.nomics.world/v22/series/OECD/KEI?dimensions={quote(geo_filter)}&observations=true"

        with APICallLogger(
            self.logger, "fetching metadata from OECD (KEI)",
            geo_filter=geo_filter, meta_url=meta_url):
            try:
                response = requests.get(meta_url)
                response.raise_for_status()
                metadata = response.json()['series']['docs']
                self.logger.info(
                    f"Found {len(metadata)} series in OECD KEI")
            except Exception as e:
                self.logger.error(f"Failed to fetch OECD metadata: {e}")
                metadata = []

        # Identify highest frequency series for each indicator
        if metadata:
            freq_rank = {'M': 3, 'Q': 2, 'A': 1}
            best_series = {}

            for item in metadata:
                dims = item['dimensions']
                key = (dims['LOCATION'],
                       dims['SUBJECT'],
                       dims['MEASURE'])
                freq = dims['FREQUENCY']
                if key not in best_series or freq_rank.get(
                    freq, 0) > freq_rank.get(
                        best_series[key]['dimensions']['FREQUENCY'], 0):
                    best_series[key] = item

            self.logger.info(
                f"Filtered {len(best_series)} unique indicators by highest frequency")
            
            predictors = pd.DataFrame(list(best_series.values()))
            predictors = predictors[['series_code', 'series_name', 'dimensions']]
            
            # Remove duplicates
            initial_count = len(predictors)
            predictors.drop_duplicates(subset=['series_code', 'series_name'],
                                   keep='first', inplace=True)
            predictors.reset_index(drop=True, inplace=True)
            
            self.logger.info(
                f"Removed {initial_count - len(predictors)} duplicates, "
                f"final count: {len(predictors)} indicators")
            
            return predictors
        
    def download_series_data(self,
                             variables: pd.DataFrame = None) -> pd.DataFrame:
        """
        Download time series data for classified variables.
        
        This method reads a CSV file containing variable series codes,
        fetches the time series data for each variable from the OECD API
        via DB.nomics, and merges them into a master DataFrame. It uses 
        the APICallLogger for detailed logging of API calls and handles errors.
        
        Parameters:
        -----------
        variables: pd.DataFrame
            Containing variable information. If None, loads from metadata file.
                
        Returns:
        --------
        pd.DataFrame
            Containing downloaded time series data.
        """
        if variables is None or variables.empty:
            variables_file = self._get_metadata_file()
            variables = pd.read_csv(variables_file)
            self.logger.info(
                f"Loaded {len(variables)} variables from {variables_file}")
        
        master_data = pd.DataFrame()
        successful_downloads = 0
        failed_downloads = 0

        all_data = dict()

        for series_code in variables['series_code']:
            # Build API URL for individual series data
            series_url = f"https://api.db.nomics.world/v22/series/OECD/KEI/{series_code}?observations=true"

            try:
                response = requests.get(series_url)
                response.raise_for_status()
                series_data = response.json()
            
                # Extract observations
                if response.status_code == 200 and 'series' in series_data:
                    
                    # Extract country, periods, and series values
                    country = series_data['dataset'][
                        'dimensions_values_labels']['LOCATION'][
                            series_data['series']['docs'][0][
                                'dimensions']['LOCATION']]
                    periods = series_data['series']['docs'][0][
                        'period_start_day']
                    colname = series_data['series']['docs'][0][
                        'dimensions']['SUBJECT'] + "." + series_data[
                            'series']['docs'][0]['dimensions'][
                                'MEASURE'] + "." + series_data['series'][
                                    'docs'][0]['dimensions']['FREQUENCY'] 
                    values = series_data['series']['docs'][0]['value']

                    # Create DataFrame for this series
                    df = pd.DataFrame({colname: values}, 
                                        index=pd.MultiIndex.from_arrays(
                                            [[country]*len(periods), periods],
                                            names=['Country', 'Period'])).reset_index()

                    if df.columns[-1] not in all_data.keys():
                        all_data[df.columns[-1]] = [df]
                    else:
                        all_data[df.columns[-1]].append(df)

                    successful_downloads += 1
                
            except Exception as e:
                failed_downloads += 1
                continue
        
        # Merge by series data
        for key, df_list in all_data.items():
            if not df_list:
                continue
            combined_df = pd.concat(df_list, ignore_index=True)
            all_data[key] = combined_df

        # Merge all series data into master DataFrame
        all_combinations = pd.DataFrame()
        for key, df in all_data.items():
            combinations = df[['Country', 'Period']].drop_duplicates()
            all_combinations = pd.concat(
                [all_combinations, combinations], ignore_index=True)
            
            all_combinations = all_combinations\
                .drop_duplicates().reset_index(drop=True)
            
        master_data = all_combinations.copy()
        # Fill values from each DataFrame
        for key, df in all_data.items():
            master_data = master_data.merge(df, on=['Country', 'Period'], how='outer')
        master_data = master_data.rename(columns={'Period': 'Date'})

        # Log final statistics
        self.logger.info(
            f"Data download summary: "
            f"{successful_downloads + failed_downloads} attempted, "
            f"{successful_downloads} successful, "
            f"{failed_downloads} failed")
        
        return master_data
    
    def run_data_pipeline(self, save_metadata=True, upload=True):
        """
        Execute the complete data download and processing pipeline for OECD data.
        
        This method handles developed and emerging markets separately.
        
        Parameters:
        -----------
        save_metadata: bool
            Whether to save intermediate metadata files.
        upload: bool
            Whether to upload the final dataset to Hugging Face Hub.

        Returns:
        --------
        Tuple
            Containing the final processed datasets (developed_data, emerging_data).
        """
        self.logger.info(f"Starting {self.data_tag} download pipeline.")

        try:
            # Step 1: Fetch indicators for developed and emerging markets separately
            developed_indicators = self.fetch_indicators(self.developed_markets)
            emerging_indicators = self.fetch_indicators(self.emerging_markets)
            
            if save_metadata:
                # Save combined indicators file
                all_indicators = pd.concat([developed_indicators, emerging_indicators], ignore_index=True)
                all_indicators.to_csv(
                    f'{root_dir}/data/{self._get_file_prefix()}_variables.csv',
                    index=False)
            
            # Step 2: Classify variables (uses base class method)
            developed_predictors = self.classify_and_save_variables(developed_indicators, save_metadata)
            emerging_predictors = self.classify_and_save_variables(emerging_indicators, save_metadata)

            # Step 3: Download series data
            developed_data = self.download_series_data(developed_predictors)
            emerging_data = self.download_series_data(emerging_predictors)

            # Step 3B (OECD Only): Update metadata file
            if save_metadata:
                combined_predictors = pd.concat([developed_predictors, emerging_predictors], ignore_index=True)
                self._consolidate_metadata(combined_predictors, save_metadata)

            # Step 4: Upload to Hugging Face
            if upload:
                repo_name=f"{username}/{self._get_file_prefix()}-download"
                data = developed_data.set_index('Date')
                upload_to_huggingface(data, repo_name, config_name="developed")
                data = emerging_data.set_index('Date')
                upload_to_huggingface(data, repo_name, config_name="emerging")

            self.logger.info(
                f"{self.data_tag} download pipeline completed. Final dataset shape: Developed = {developed_data.shape}, Emerging = {emerging_data.shape}")
            
            return developed_data, emerging_data

        except Exception as e:
            self.logger.error(f"{self.data_tag} download pipeline failed: {e}")
            raise
    
    def _consolidate_metadata(self, predictors: pd.DataFrame, save_metadata: bool = True) -> None:
        """
        Consolidate metadata by grouping by generic series codes.
        
        Parameters:
        -----------
        predictors: pd.DataFrame
            Containing predictor variables to consolidate.
        """
        df = predictors.copy()

        # Create generic series_code
        def create_generic_series_code(row):
            try:
                dimensions = ast.literal_eval(row['dimensions'])
                subject = dimensions.get('SUBJECT', '')
                measure = dimensions.get('MEASURE', '')
                frequency = dimensions.get('FREQUENCY', '')
                
                generic_code = f"{subject}.{measure}.{frequency}"
                return generic_code
            except: # Format: SUBJECT.COUNTRY.MEASURE.FREQUENCY
                parts = row['series_code'].split('.')
                if len(parts) >= 4:
                    return f"{parts[0]}.{parts[2]}.{parts[3]}"
                return row['series_code']
        
        df['generic_series_code'] = df.apply(create_generic_series_code, axis=1)
        
        # Create generic series_name
        def clean_series_name(name):
            name = name.split('–')[0].strip()
            return name
        
        df['cleaned_series_name'] = df['series_name'].apply(clean_series_name)
        
        # Create generic dimensions
        def clean_dimensions(dimensions_str):
            try:
                dimensions = ast.literal_eval(dimensions_str)
                if 'LOCATION' in dimensions:
                    del dimensions['LOCATION']
                return str(dimensions)
            except:
                return dimensions_str
        
        df['cleaned_dimensions'] = df['dimensions'].apply(clean_dimensions)
        
        # Group by generic series code and take the first occurrence
        result = df.groupby('generic_series_code').first().reset_index()
        
        # Create final dataframe with desired columns
        final_df = pd.DataFrame({
            'series_code': result['generic_series_code'],
            'series_name': result['cleaned_series_name'],
            'dimensions': result['cleaned_dimensions'],
            'bucket': result['bucket'],
            'bucket_name': result['bucket_name']
        })
        
        # Update metadata file
        if save_metadata:
            final_df.to_csv(
                        f'{root_dir}/data/{self._get_file_prefix()}_variables.csv',
                        index=False)

      
class YahooFinanceDataLoader(DataLoader):
    """
    Yahoo Finance specific data loader that inherits from DataLoader.

    This class implements methods to fetch indicators, download series
    data, classify variables, and run the full data processing pipeline.

    Attributes:
    -----------
    config_path (str)
        Path to the configuration file.
    topics (List[str])
        List of topics to filter indicators.
    sources (List[str])
        List of data sources to fetch indicators.
    developed_markets (List[str])
        List of developed countries for data filtering.
    emerging_markets (List[str])
        List of emerging markets for data filtering.
    """

    def __init__(self, data_tag: str = "YahooFinance", config_path: str = "config.json"):
        """
        Initialize YahooFinanceDataLoader with configuration.

        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.
        """
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")
        self.data_tag = data_tag.upper()
        self.config = self._load_config(config_path)

        # Extract Yahoo Finance specific configuration
        self.tickers = self.config["YAHOO_FINANCE_TICKERS"]

        self.logger.info("YahooFinanceDataLoader initialized successfully")

    def fetch_indicators(self) -> List[str]:
        """
        Fetch indicators from Yahoo Finance API.
        
        This method retrieves the list of tickers from the configuration.
        
        Returns:
        --------
        pd.DataFrame
            Containing the list of tickers.
        """
        if not self.tickers:
            self.logger.error("No tickers found in configuration")
            raise ValueError("No tickers found in configuration")
        
        if isinstance(self.tickers, str):
            indicators = [self.tickers]
        elif isinstance(self.tickers, dict):
            indicators = list(self.tickers.values())
            indicators = [t for ticker in indicators for t in ticker]
        else:
            indicators = self.tickers
        return indicators
    
    def download_series_data(self, variables: List[str] = None) -> pd.DataFrame:
        """
        Download time series data for specified tickers.
        
        This method fetches historical stock data for each ticker using
        the yfinance library and merges them into a master DataFrame.
        
        Parameters:
        -----------
        variables: List[str]
            List of tickers to download data for.
            If None, uses the tickers from the configuration.
                
        Returns:
        --------
        pd.DataFrame
            Containing downloaded time series data.
        """
        if variables is None or not variables:
            variables = self.fetch_indicators()

        self.logger.info(
            f"Downloading data for {len(variables)} tickers.")
        data = yf.download(variables, start="1945-01-01", end="2025-01-01")
        vars = data.columns.get_level_values(0).unique().tolist()
        vars.extend(["52WeekHigh", "52WeekLow", "NewHigh", "NewLow", "HighLowDiff", "CloseOpenDiff", "Volatility"])
        columns = [ticker + "." + var for ticker in variables for var in vars]

        save_df = pd.DataFrame(index = data.index, columns=columns)
        for ticker in variables:
            for var in vars:
                if (var, ticker) in data.columns.tolist():
                    save_df[ticker + "." + var] = data[(var, ticker)]
            save_df[ticker + ".52WeekHigh"] = save_df[ticker + ".Close"].rolling(window=252, min_periods=1).max()
            save_df[ticker + ".52WeekLow"] = save_df[ticker + ".Close"].rolling(window=252, min_periods=1).min()
            save_df[ticker + ".NewHigh"] = save_df[ticker + ".Close"] >= save_df[ticker + ".52WeekHigh"]
            save_df[ticker + ".NewLow"] = save_df[ticker + ".Close"] <= save_df[ticker + ".52WeekLow"]
            save_df[ticker + ".HighLowDiff"] = save_df[ticker + ".High"] - save_df[ticker + ".Low"]
            save_df[ticker + ".CloseOpenDiff"] = save_df[ticker + ".Close"] - save_df[ticker + ".Open"]
            save_df[ticker + ".Volatility"] = save_df[ticker + ".Close"].pct_change().rolling(window=21).std() * np.sqrt(252)
            
        save_df = save_df.ffill()
        save_df = save_df.reset_index()

        if save_df.empty:
            self.logger.error("No data downloaded, check ticker symbols")
            return pd.DataFrame()
        return save_df
    
    def run_data_pipeline(self, save_metadata=True, upload=True) -> pd.DataFrame:
        """
        Execute the complete data download and processing pipeline for Yahoo Finance data.
        
        Parameters:
        -----------
        save_metadata: bool
            Whether to save intermediate metadata files.
        upload: bool
            Whether to upload the final dataset to Hugging Face Hub.

        Returns:
        --------
        pd.DataFrame
            Containing the final processed dataset.
        """
        self.logger.info(f"Starting {self.data_tag} download pipeline.")

        try:
            # Step 1: Fetch indicators
            indicators = self.fetch_indicators()
            if save_metadata:
                df = pd.DataFrame({'Tickers': indicators})
                df.to_csv(
                    f'{root_dir}/data/{self._get_file_prefix()}_variables.csv',
                    index=False)
            
            # Step 2: Download series data
            master_data = self.download_series_data(indicators)
            
            # Step 3: Upload to Hugging Face
            if upload:
                data = master_data.set_index('Date')
                repo_name=f"{username}/{self._get_file_prefix()}-download"
                upload_to_huggingface(data, repo_name, config_name='train')
            
            self.logger.info(
                f"{self.data_tag} download pipeline completed. Final dataset shape: {master_data.shape}")
            return master_data
            
        except Exception as e:
            self.logger.error(f"{self.data_tag} download pipeline failed: {e}")
            raise

class JSTDataLoader(DataLoader):
    """
    Jordà-Schularick-Taylor Macrohistory Database specific data loader that inherits from DataLoader.
    
    This class implements methods to fetch indicators, download series
    data, classify variables, and run the full data processing pipeline.

    Attributes:
    -----------
    config_path (str)
        Path to the configuration file.
    topics (List[str])
        List of topics to filter indicators.
    sources (List[str])
        List of data sources to fetch indicators.
    developed_markets (List[str])
        List of developed countries for data filtering.
    emerging_markets (List[str])
        List of emerging markets for data filtering.
    """

    def __init__(self, data_tag: str = "JST", config_path: str = "config.json"):
        """
        Initialize JSTDataLoader with configuration.

        Parameters:
        -----------
        config_path: str
            Path to the JSON configuration file.
        """
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")
        self.data_tag = data_tag.upper()
        self.data = self.download_series_data()

        self.logger.info("JSTDataLoader initialized successfully")

    def fetch_indicators(self) -> List[str]:
        """
        Fetch indicators from Yahoo Finance API.
        
        This method retrieves the list of variables from the configuration.
        
        Returns:
        --------
        pd.DataFrame
            Containing the list of variables.
        """
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            self.logger.error("No data found in configuration")
            raise ValueError("No data found in configuration")
        
        columns = [col for col in self.data.columns if col not in ['Country', 'Year', 'Date']]
        return pd.DataFrame(columns)

    def download_series_data(self, variables = None):
        data = pd.read_excel(
            f"{root_dir}/data/JSTdatasetR6.xlsx", sheet_name="Sheet1")
        
        # Skip crisis label columns
        skip_columns = ['crisisJST', 'crisisJST_old']
        data = data.drop(columns=skip_columns, errors='ignore')

        # Rename columns for consistency
        data.rename(columns={'year': 'Date', 'country': 'Country'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y')

        # Rename USA and United States and UK and United Kingdom
        data.loc[data['Country'] == "USA", 'Country'] = "United States"
        data.loc[data['Country'] == "UK", 'Country'] = "United Kingdom"

        return data

    def run_data_pipeline(self, save_metadata=True, upload=True):
        """
        Execute the complete data download and processing pipeline.
        
        This method orchestrates the entire workflow:
        1. Fetch indicators
        2. Upload the final dataset to Hugging Face Hub.
        
        Parameters:
        -----------
        save_metadata: bool
            Whether to save intermediate metadata files.
        upload: bool
            Whether to upload the final dataset to Hugging Face Hub.

        Returns:
        --------
        Tuple
            Containing the final processed datasets (developed_data, emerging_data).
            
        Raises:
        -------
        ValueError: If required environment variables are not set.
        Exception: If any step in the pipeline fails.
        """
        self.logger.info(f"Starting {self.data_tag} download pipeline.")

        try:
            # Step 1: Fetch indicators
            indicators = self.fetch_indicators()
            if save_metadata:
                indicators.to_csv(
                    f'{root_dir}/data/{self._get_file_prefix()}_variables.csv',
                    index=False)
            
            # Step 2: Upload to Hugging Face
            if upload:
                repo_name = f"{username}/{self._get_file_prefix()}-download"
                developed = self.data.set_index('Date')
                upload_to_huggingface(developed, repo_name, config_name="developed")

            self.logger.info(
                f"{self.data_tag} download pipeline completed. Final dataset shape: Developed = {developed.shape}, Emerging = N/A")

            return developed

        except Exception as e:
            self.logger.error(f"{self.data_tag} download pipeline failed: {e}")
            raise

class CrisisLabeller:
    """
    Class for labeling crisis periods in financial time series data.
    
    This class processes crisis data to create labels for machine learning
    models, marking periods before crises as positive examples and handling
    recovery periods appropriately.
    
    Attributes:
    -----------
    logger: logger.Logger
        Logger instance for logging messages.
    filepath (str)
        Path to the crisis data file.
    df (pd.DataFrame)
        DataFrame containing crisis data.
    crisis_types (List[str])
        List of crisis types to consider.
    """
    
    def __init__(self, filepath: str,
                 sheetname: str = "Crisis",
                 crisis_types: List[str] = None):
        """
        Initialize the CrisisLabeller with a file path.

        Parameters:
        -----------
        filepath: str
            Path to the crisis data Excel file.
        sheetname: str
            Name of the sheet to read from the Excel file.
        crisis_types: List[str]
            List of crisis types to filter the data.
            If None, all crisis types will be used.
                
        Raises:
        -------
        FileNotFoundError: If the specified file is not found.
        """
        self.logger = setup_logging(f"{self.__class__.__name__}_logs.txt")
        self.filepath = filepath
        self.df = pd.read_excel(filepath, sheet_name = sheetname)

        if crisis_types is not None:
            self.crisis_types = crisis_types
        else:
            self.crisis_types = self.df.columns[
                self.df.columns.str.contains("crisis|crises", case=False)].tolist()
        self.df = self.mark_crises(self.df)
        
        self.logger.info(
            f"CrisisDataLoader initialized with {len(self.crisis_types)} crisis types")
        
    def mark_crises(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark crisis periods in the DataFrame.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Containing time series data with 'Country' and 'Year' columns.
        
        Returns:
        --------
        pd.DataFrame
            With additional boolean 'is_crisis' column.
        """

        for crisis in self.crisis_types:
            if crisis not in df.columns:
                self.logger.warning(
                    f"Crisis type '{crisis}' not found in DataFrame")
                continue
            
        df['is_crisis'] = df[self.crisis_types].apply(
            lambda row: any(x == 1.0 for x in row), axis=1)
        
        self.logger.info(f"Created boolean is_crisis column in DataFrame")
        return df
    
    def create_labels(self,
                      lookback_years: int=2,
                      recovery_years: int=2,
                      output_file: str = None) -> pd.DataFrame:
        """
        Create labels for crisis periods in the DataFrame.

        Parameters:
        -----------
        lookback_years: int
            Number of years before crisis to mark as positive labels.
        recovery_years: int
            Number of years after crisis to mark for dropping.
        output_file: str
            Name of the output CSV file for labels.

        Returns:
        --------
        pd.DataFrame
            DataFrame with 'Country' and 'Year' columns for positive crisis labels.
            
        Raises:
        -------
        ValueError: If 'is_crisis' column is not found in DataFrame.
        """
        if 'is_crisis' not in self.df.columns:
            self.logger.error("is_crisis column not found in DataFrame")
            raise ValueError("is_crisis column not found in DataFrame")
        
        self.df = self.df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
        self.df['label'] = 0
        self.df = self.df.groupby('Country', group_keys=False).apply(
            lambda x: self.process_country(x,
                                           crisis_types = self.crisis_types,
                                           lookback_years=lookback_years,
                                           recovery_years=recovery_years),
        )
        
        # Extract positive labels (country, year) pairs
        labels = self.df[
            (self.df['label'] == 1) & (~self.df['drop'])][['Country', 'Year']]

        if 'Year' in labels.columns:
            labels['Year'] = labels['Year'].astype(int)
        labels = labels.sort_values(by=['Country', 'Year']).reset_index(drop=True)
        
        if output_file is not None:
            labels.to_csv(output_file, index=False)
            self.logger.info(f"Created crisis labels and saved to {output_file}")
        else:
            self.logger.info(f"Created crisis labels")
        return labels
    
    @staticmethod
    def process_country(group, crisis_types, lookback_years=2, recovery_years=2):
        """
        Process crisis labeling for a single country.
        
        For each country:
        - Set label=1 for lookback_years before a crisis
        - Mark recovery_years after crisis end for dropping
        
        Parameters:
        -----------
        group: DataFrame group
            For a single country.
        lookback_years: int
            Number of years before crisis to label.
        recovery_years: int
            Number of years after crisis to mark for dropping.
            
        Returns:
        --------
        Modified group DataFrame with 'label' and 'drop' columns.
        """
        years = group['Year'].values
        is_crisis = group[crisis_types].apply(lambda row: any(x == 1.0 for x in row), axis=1).values
        label = np.zeros(len(group))
        drop_idx = set()
        
        for i in range(len(years)):
            if is_crisis[i]:
                # Set label=1 for specified years before crisis
                for offset in range(1, lookback_years + 1):
                    idx = i - offset
                    if idx >= 0 and not is_crisis[idx]:
                        label[idx] = 1
                
                # Mark specified years after crisis for dropping
                for offset in range(1, recovery_years + 1):
                    idx = i + offset
                    if idx < len(years):
                        drop_idx.add(idx)
        
        group = group.copy()
        group['label'] = label
        group['drop'] = False
        
        if drop_idx:
            group.iloc[list(drop_idx), group.columns.get_loc('drop')] = True
        
        return group