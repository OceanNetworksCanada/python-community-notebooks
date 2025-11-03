"""
This .py file contains functions which may be commonly used in the ONC python-community-notebooks repository.

There is no strict format or documentation requirement.
It is strongly encouraged that you add typing hints and descriptive comments

"""

from datetime import datetime, timedelta
from geopy import distance
from netrc import netrc
import numpy as np
from os import PathLike
import pandas as pd
import re
import xarray as xr



FlagTerm = 'qaqc_flag'  # String to prepend to flag variables.

# Approximate locations for ONC BCF Terminals.
class BCFTerminal:
    class Tsawwassen:
        latitude: float = 49.006621
        longitude: float = -123.132309

    class DukePoint:
        latitude: float = 49.162529
        longitude: float = -123.891036

    class DepartureBay:
        latitude: float = 49.193512
        longitude: float = -123.954777

    class Gabriola:
        latitude: float = 49.177846
        longitude: float = -123.858655

    class NanaimoHarbor:
        latitude: float = 49.166714
        longitude: float = -123.930933

    class SwartzBay:
        latitude: float = 48.689047
        longitude: float = -123.410817

    class PortMcneil:
        latitude: float = 50.592621
        longitude: float = -127.085620

    class AlertBay:
        latitude: float = 50.587972
        longitude: float = -126.931313

    class Sointula:
        latitude: float = 50.626701
        longitude: float = -127.018700

    class HorseshoeBay:
        latitude: float = 49.375791
        longitude: float = -123.271643



def format_datetime(dt: datetime | str) -> str:
    """
    Format an incoming datetime representation to a format that is compatible
        with the ONC REST API. If None is provided, then the API will default
        to using the tail end of the available data.

    :param dt: A datetime object, string representation of a date, or None.
    :return: A string in the format of 'YYYY-mm-ddTHH:MM:SS.fffZ'.
    """
    if dt is None:
        return None
    else:
        dt = pd.to_datetime(dt)
        dtstr = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        return dtstr


def get_onc_token_from_netrc(netrc_path: PathLike | None = None,
                             machine: str = 'data.oceannetworks.ca') -> str:
    """
    Retrieve an Oceans 3.0 API token from a .netrc file.

    :param netrc_path: Path to a .netrc file. If None, the user directory is assumed.
    :param machine: The machine lookup name in the .netrc file. Default is
                    'data.oceannetworks.ca'.
    :return: An Oceans 3.0 API token.
    """
    if netrc_path is None:
        _, __, onc_token = netrc().authenticators(machine)
    else:
        _, __, onc_token = netrc(netrc_path).authenticators(machine)
    return onc_token


def scrub_token(query_url: str) -> str:
    """
    Replace a token in a query URL with the string 'REDACTED' so that users don't
    accidentally commit their tokens to public repositories if ONC Info/Warnings are
    too verbose.

    :param query_url: An Oceans 3.0 API URL with a token query parameter.
    :return: A scrubbed url.
    """
    token_regex = r'(&token=[a-f0-9-]{36})'
    token_qp = re.findall(token_regex, query_url)[0]
    redacted_url = query_url.replace(token_qp, '&token=REDACTED')
    return redacted_url


def var_name_from_sensor_name(sensor_name: str) -> str:
    """
    Create a new variable name from a sensorName. The sensorName is generally
        more descriptive, but contains spaces and parentheses which is not ideal for
        packages that support dot indexing for data access, so this function will
        remove them.

    :param sensor_name: The sensorName attribute from a json response.
    :return: An updated variable name.
    """
    var_name = sensor_name.replace(' ', '_').lower()
    var_name = var_name.replace('(', '')
    var_name = var_name.replace(')', '')
    return var_name


def json_var_data_to_dataframe(var_data: dict) -> pd.DataFrame:
    """
    Convert a single variable's data from a json response to a pandas DataFrame.

    :param var_data: Pulled from a subset of the sensorData section
        of a json response.
    :return: A pandas DataFrame.
    """
    var_name = var_name_from_sensor_name(var_data['sensorName'])
    flag_var_name = '_'.join((FlagTerm, var_name))
    var_times = var_data['data']['sampleTimes']
    var_values = var_data['data']['values']
    var_flags = var_data['data']['qaqcFlags']
    vdf = pd.DataFrame({'time': var_times,
                        var_name: var_values,
                        flag_var_name: var_flags})

    vdf['time'] = pd.to_datetime(vdf['time']).dt.tz_localize(None)
    vdf['time'] = vdf['time'].astype('datetime64[ms]')
    vdf.index = vdf['time']
    vdf = vdf.drop(columns=['time'])
    var_metadata = {k: v for k, v in var_data.items() if
                    k not in ['actualSamples', 'data', 'outputFormat']}
    return (vdf, var_metadata)



def convert_scalar_data(json_response_data: dict,
                 out_as: str ='xarray',
                 scrub_url: bool = True) -> pd.DataFrame | xr.Dataset | dict:
    """
    Convert a full json response containing scalar data to a pandas DataFrame
        or xarray Dataset.

    :param json_response_data: A json response from a scalarData endpoint.
    :param out_as: 'json', 'pandas', or 'xarray'.
    :param scrub_url: If True, the token is removed from the query url when
        a UserWarning is raised.
    :return:
    """
    qaqc_flag_info = json_response_data['qaqcFlagInfo']
    qaqc_flag_info = '\n'.join(
        [':'.join((k, v)) for k, v in qaqc_flag_info.items()])

    if 'metadata' in json_response_data.keys():
        dev_cat_code = json_response_data['metadata']['deviceCategoryCode']
        loc_name = json_response_data['metadata']['locationName']
    else:
        dev_cat_code = json_response_data['parameters']['deviceCategoryCode']
        loc_name = 'Metadata query input set to minimum. | Not Found'

    loc_code = json_response_data['parameters']['locationCode']
    sensor_data = json_response_data['sensorData']

    if sensor_data is None:
        if scrub_url is True:
            query_url = scrub_token(json_response_data['queryUrl'])
        else:
            query_url = json_response_data['queryUrl']
        raise UserWarning(f"No data found for request: {query_url}")

    dfs, var_metadata = zip(*[json_var_data_to_dataframe(vd)
                              for vd in sensor_data])
    df = pd.concat(dfs, axis=1)

    if out_as == 'pandas':
        out = df
        vars = out.columns
    elif out_as == 'xarray':
        out = df.to_xarray()
        vars = out.data_vars

    for vmd in var_metadata:
        var_name = var_name_from_sensor_name(vmd['sensorName'])
        out[var_name].attrs = vmd
        out[var_name].attrs['deviceCategoryCode'] = dev_cat_code
        out[var_name].attrs['locationName'] = loc_name
        out[var_name].attrs['locationCode'] = loc_code

        flag_var_name = '_'.join((FlagTerm, var_name))
        if flag_var_name in vars:
            out[flag_var_name].attrs['ancillary_variable'] = var_name
            out[flag_var_name].attrs['flag_meanings'] = qaqc_flag_info

    out.attrs['deviceCategoryCode'] = dev_cat_code
    out.attrs['locationName'] = loc_name
    out.attrs['locationCode'] = loc_code
    out.attrs['qaqcFlagInfo'] = qaqc_flag_info
    return out



def split_periods(da: xr.DataArray, min_gap: int = 60 * 5) -> list[dict]:
    """
    Split a time series into periods of time containing consecutive data with a
        specified minimum gap in between each period.

    :param da: A time-indexed xarray DataArray.
    :param min_gap: The minimum number of seconds between data points that constitutes
        a break in the time series.
    :return: A list of dictionaries with 'dateFrom' and 'dateTo' keys for each period.
    """

    # First sort the data by time if it isn't already sorted.
    da = da.sortby('time')

    dts = list(da.where(da['time'].diff('time') >
                             np.timedelta64(min_gap, 's'), drop=True).get_index('time'))

    if da.time.min() != dts[0]:
        dts = [pd.to_datetime(da.time.min().values)] + dts

    periods = []
    for dt in dts:
        if dt == dts[-1]:
            start = dt
            stop = None
        else:
            dtidx = dts.index(dt)
            start = dt
            stop = dts[dtidx + 1] - timedelta(seconds=30)
        period = da.sel(time=slice(start, stop))
        if len(period.time.values) == 0:
            continue
        else:
            _p = {'dateFrom': pd.to_datetime(period.time.min().values),
                  'dateTo': pd.to_datetime(period.time.max().values)}
            periods.append(_p)
    if len(periods) == 0:
        _p = {'dateFrom': pd.to_datetime(da.time.min().values),
              'dateTo': pd.to_datetime(da.time.max().values)}
        periods = [_p]
    return periods



def identify_transit_ports(transit_lat: xr.DataArray,
                   transit_lon: xr.DataArray,
                   max_distance: float = 1500) -> tuple:
    """
    Identify the start and stop port based on the minimum and maximum times of the
        input dataset (transit).

    :param transit_lat: The latitude values of the transit.
    :param transit_lon: The longitude values of the transit.
    :param max_distance: The maximum allowed distance (in meters) used to identify a
        start or stop port.
    :return: A tuple indicating the (start_location, stop_location).
    """

    terminals = [k for k,v in BCFTerminal.__dict__.items() if '__' not in k]
    blat = transit_lat.where(transit_lat.time == transit_lat.time.min(), drop = True)
    blon = transit_lon.where(transit_lon.time == transit_lon.time.min(), drop = True)
    transit_start = (blat.values.tolist()[0], blon.values.tolist()[0])

    elat = transit_lat.where(transit_lat.time == transit_lat.time.max(), drop = True)
    elon = transit_lon.where(transit_lon.time == transit_lon.time.max(), drop = True)
    transit_end = (elat.values.tolist()[0], elon.values.tolist()[0])

    start_port = None
    end_port = None

    for terminal in terminals:
        loc = getattr(BCFTerminal, terminal)
        start_dist = distance.great_circle(transit_start,
                                           (loc.latitude, loc.longitude)).m
        end_dist = distance.great_circle(transit_end,
                                         (loc.latitude, loc.longitude)).m
        if start_dist < max_distance and start_port is None:
            start_port = terminal
        if end_dist < max_distance and end_port is None:
            end_port = terminal

    if start_port is None:
        start_port = 'Unknown'
    if end_port is None:
        end_port = 'Unknown'
    return (start_port, end_port)


def calculate_distance_to_known_ports(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate the distance, in meters, between a given nav point and a
        known BCF terminal.
    :param dataset: The dataset. Must have latitude and longitude variables, at minimum.
    :return: A new dataset with variables describing "distance_to_X" where X is the
        terminal of interest.
    """
    dspts = list(
        zip(dataset.latitude.values.tolist(), dataset.longitude.values.tolist()))

    terminals = [k for k, v in BCFTerminal.__dict__.items() if '__' not in k]
    for terminal in terminals:
        terminal_dists = []
        loc = getattr(BCFTerminal, terminal)
        for dspt in dspts:
            if np.any(np.isnan(dspt)):
                terminal_dists.append(np.nan)
            else:
                terminal_dists.append(np.round(
                    distance.great_circle(dspt, (loc.latitude, loc.longitude)).m))
        dataset[f"distance_to_{terminal.lower()}"] = (['time'], terminal_dists)
    return dataset


def identify_profiles(cable_length, profile_direction: str = 'all',
                      buffer: int = 10,
                      max_allowed_std: float = 0.02, min_gap: int = 180):

    """
    Identify Barkley Canyon profiles based on the cable length.

    :param cable_length: The input cable length.
    :param profile_direction: Whether to return "up", "down", or "all" profile types.
    :param buffer: The number of seconds to add to the profiler tails.
    :param max_allowed_std: The maximum allowed standard deviation (in meters)
        to consider no movement in the cable length.
    :param min_gap: The minimum number of seconds in between profiles.
    :return: A list of profile start and stop times.
    """


    flag_cl = flat_line_test(cable_length, max_allowed_std=max_allowed_std)
    profiling_state = flag_cl.where(flag_cl == 1, drop=True)
    profiles = split_periods(profiling_state, min_gap = min_gap)

    assigned_profiles = []
    for profile in profiles:
        _cl = cable_length.sel(time=slice(profile['dateFrom'], profile['dateTo']))
        _start = _cl.sel(time=_cl.time.min())
        _stop = _cl.sel(time=_cl.time.max())
        if _start - _stop > 0:
            profile_dir = 'down'
        else:
            profile_dir = 'up'

        profile['direction'] = profile_dir
        profile['dateFrom'] = pd.to_datetime(profile['dateFrom']
                                              - np.timedelta64(buffer, 's'))
        profile['dateTo'] = pd.to_datetime(profile['dateTo']
                                            + np.timedelta64(buffer, 's'))

        assigned_profiles.append(profile)

    if profile_direction == 'all':
        return assigned_profiles
    elif profile_direction == 'up':
        up_pros = [p for p in assigned_profiles if p['direction'] == 'up']
        return up_pros
    elif profile_direction == 'down':
        down_pros = [p for p in assigned_profiles if p['direction'] == 'down']
        return down_pros



## QAQC Tests

class FLAG:
    NOT_EVALUATED: int = 0
    OK: int = 1
    PROBABLY_OK: int = 2
    PROBABLY_BAD: int = 3
    BAD: int = 4
    MISSING_DATA: int = 9


def flat_line_test(data: xr.DataArray,
                   fail_window_size: int = 5,
                   suspect_window_size: int = 3,
                   max_allowed_std: float = 0) -> xr.DataArray:
    """
    Perform a modified version of the QARTOD Flat Line Test.

    :param data: The input dataset.
    :param fail_window_size: The maximum number of consecutive samples that need
        to be within a certain standard deviation to be flagged as bad.
    :param suspect_window_size: The maximum number of consecutive samples that need
        to be within a certain standard deviation to be flagged as probably bad.
    :param max_allowed_std: The maximum standard deviation within the window
        to be considered a flat line.
    :return: An xr.DataArray of flags with the same shape as the input data.
    """

    # Fail Window Construction
    wf = data.rolling({'time': fail_window_size}).construct('window')
    wf_std = wf.std(dim='window')

    # Suspect Window Construction
    ws = data.rolling({'time': suspect_window_size}).construct('window')
    ws_std = ws.std(dim='window')

    flag = xr.full_like(data, FLAG.NOT_EVALUATED, dtype='int8')
    flag = xr.where(ws_std <= max_allowed_std, FLAG.PROBABLY_BAD, flag)
    flag = xr.where(wf_std <= max_allowed_std, FLAG.BAD, flag)
    flag = xr.where(flag == FLAG.NOT_EVALUATED, FLAG.OK, flag)

    return flag