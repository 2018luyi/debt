'''
A O(nlogn) time implementation of the knockoff filter.

Author: Wesley Tansey
Date: 3/27/2020
'''
import numpy as np

def knockoff_filter(knockoff_stats, alpha, offset=1.0, is_sorted=False):
    '''Perform the knockoffs selection procedure at the target FDR threshold.
    :param knockoff_stats: vector of knockoff statistics
    :param alpha: nominal false discovery rate
    :param offset: equal to one for strict false discovery rate control
    :param is_sorted: if True, assumes knockoff_stats is already sorted.
    :return an array of indices selected at the alpha FDR level.
    This implementation runs in nlogn time.'''
    n = len(knockoff_stats) # Length of the stats array
    if is_sorted:
        order = np.arange(n)
        sorted_stats = knockoff_stats
    else:
        order = np.argsort(knockoff_stats)
        sorted_stats = knockoff_stats[order]

    # Edge case: if there are no positive stats, just return empty selections
    if sorted_stats[-1] <= 0:
        return np.array([], dtype='int64')

    ridx = np.searchsorted(sorted_stats, 0, side='right') # find smallest positive value index
    lidx = np.searchsorted(sorted_stats, -sorted_stats[ridx], side='left') # find matching negative value index

    # Numpy correction: if -sorted_stats[ridx] is less than any number in the list,
    # searchsorted returns 0 instead of -1. This is not the desired behavior here.
    if lidx == 0 and sorted_stats[lidx] >= -sorted_stats[ridx]:
        # If the current ratio isn't good enough, it will never get better.
        if (lidx + 1 + offset) / max(1, n - ridx) > alpha:
            return np.array([], dtype='int64')

        # If we're below the alpha threshold, return everything positive
        return order[ridx:]

    # If the knockoff ratio is below the threshold, return all stats
    # at or above the current value
    while (lidx + 1 + offset) / max(1, n - ridx) > alpha:
        # If we were at the end of the negative values, we won't get
        # a better ratio by going further down the positive value side.
        if lidx == -1:
            return np.array([], dtype='int64')

        # Move to the next smallest positive value
        ridx += 1

        # Check if we've reached the end of the list
        if ridx == n:
            break

        # Find the matching negative value
        while lidx >= 0 and sorted_stats[lidx] > -sorted_stats[ridx]:
            lidx -= 1

    # Return the set of stats with values above the threshold
    return order[ridx:]

