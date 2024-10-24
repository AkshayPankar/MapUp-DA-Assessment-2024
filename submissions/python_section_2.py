import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    locations = pd.concat([df['location_A'], df['location_B']]).unique()
    locations.sort()

   
    distance_matrix = pd.DataFrame(
        data=np.inf, 
        index=locations, 
        columns=locations
    )

    # Set diagonal to zero as distance from a point to itself is zero
    np.fill_diagonal(distance_matrix.values, 0)

  
    for _, row in df.iterrows():
        loc_A = row['location_A']
        loc_B = row['location_B']
        dist = row['distance']
        distance_matrix.loc[loc_A, loc_B] = dist
        distance_matrix.loc[loc_B, loc_A] = dist  

    # Use the Floyd-Warshall algorithm to calculate the shortest paths
    for k in locations:
        for i in locations:
            for j in locations:
                # Update the distance if a shorter path is found through intermediate 'k'
                distance_matrix.loc[i, j] = min(
                    distance_matrix.loc[i, j], 
                    distance_matrix.loc[i, k] + distance_matrix.loc[k, j]
                )

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    df = df.reset_index()
    df = df.melt(id_vars='index', var_name='id_end', value_name='distance')
    df = df.rename(columns={'index': 'id_start'})

    # Filter out rows where id_start is the same as id_end (no self-distances)
    df = df[df['id_start'] != df['id_end']].reset_index(drop=True)

    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
   
    reference_distances = df[df['id_start'] == reference_id]['distance']
    reference_avg = reference_distances.mean()

    # Calculate the 10% threshold bounds
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

   
    avg_distances = df.groupby('id_start')['distance'].mean()

    # Filter ids whose average distance lies within the threshold range
    matching_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index

   
    return sorted(matching_ids)

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
     # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.05,
        'car': 0.1,
        'rv': 0.15,
        'bus': 0.2,
        'truck': 0.25
    }
    
    # Calculate toll rates by multiplying distance with each coefficient
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
