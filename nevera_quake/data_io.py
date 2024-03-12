from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
import datetime as dt

def load_stream(path: str) -> 'obspy.core.stream.Stream':
    """Load a Stream object from a file.

    Args:
        path (str): Path to the input file. Supported formats can be found at:
                    http://docs.obspy.org/tutorial/code_snippets/reading_seismograms.html

    Returns:
        obspy.core.stream.Stream: A merged Stream object.
    """
    try:
        stream = read(path)
        stream.merge()
    except Exception as e:
        # Handle exceptions such as file not found, unsupported format, etc.
        print(f"Failed to load stream from {path}: {e}")
        raise
    return stream

def load_catalog(path: str) -> pd.DataFrame:
    """Load an event catalog from a CSV file.

    Args:
        path (str): Path to the input .csv file.

    Returns:
        pd.DataFrame: A DataFrame containing the catalog.
    """
    try:
        catalog = pd.read_csv(path)
        if 'utc_timestamp' not in catalog.columns:
            catalog['utc_timestamp'] = pd.to_datetime(catalog['origintime']).apply(lambda x: UTCDateTime(x).timestamp)
    except Exception as e:
        print(f"Failed to load catalog from {path}: {e}")
        raise
    return catalog

def write_stream(stream: 'obspy.core.stream.Stream', path: str) -> None:
    """Write a Stream object to a file.

    Args:
        stream (obspy.core.stream.Stream): The Stream object to write.
        path (str): The output file path.
    """
    try:
        stream.write(path, format='MSEED')
    except Exception as e:
        print(f"Failed to write stream to {path}: {e}")
        raise

def write_catalog(events: list, path: str) -> None:
    """Write an event catalog to a CSV file.

    Args:
        events (list): A list of UTCDateTime objects.
        path (str): The output file path.
    """
    catalog = pd.DataFrame({'utc_timestamp': [t.timestamp for t in events]})
    catalog.to_csv(path, index=False)

def write_catalog_with_clusters(events: list, clusters: list, latitudes: list, longitudes: list, depths: list, path: str) -> None:
    """Write an event catalog with cluster information to a CSV file.

    Args:
        events (list): A list of timestamps.
        clusters (list): A list of cluster IDs.
        latitudes (list), longitudes (list), depths (list): Location and depth data.
        path (str): The output file path.
    """
    catalog = pd.DataFrame({
        'utc_timestamp': events,
        'cluster_id': clusters,
        'latitude': latitudes,
        'longitude': longitudes,
        'depth': depths
    })
    catalog.to_csv(path, index=False)
