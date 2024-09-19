"""
https://www.geeksforgeeks.org/merging-intervals/
"""
import numpy as np

def mergeIntervals(intervals, PRINT=False):
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
 
    if PRINT:
        print("The Merged Intervals are :", end=" ")
        for i in range(len(stack)):
            print(stack[i], end=" ")
    return stack

def excludeIntervals_int(list1, list2):
    """
    # exclude intervals in list2 from list1.
    this assumes each interval to have the smaller number first, each list to have the intervals ordered ascending and only integer entries.
    e.g. list1 = [(0,10),(15,20)]; list2 = [(2,3),(5,6)]
    res = [(0, 2), (3, 5), (6, 10), (15, 20)]
    """
    s = min(list1[0][0], list2[0][0])
    e = max(list1[-1][-1], list2[-1][-1])
    # by default no slot is in an interval (0)
    arr = np.full((e - s + 3,), 0)
    # mark intervals from list1
    for a, b in list1:
        arr[a-s+1:b-s+1] = 1
    # unmark intervals from list2
    for a, b in list2:
        arr[a-s+1:b-s+1] = 0
    # detect edges (start and end points)
    starts = np.where((arr[1:] == 1) & (arr[:-1] == 0))[0] + s
    ends = np.where((arr[:-1] == 1) & (arr[1:] == 0))[0] + s
    res = list(zip(starts, ends))
    return res

def locateInIntervals(list1, x):
    """
    find the interval in list1 x belongs to, and the interval before x
    assume list1 is sorted and contains intervals that do not overlap
    return: [ibefore, i_in]
    """
    starts = [x[0] for x in list1]
    if x < starts[0]: 
        return [-42, -42] # x lies before the first interval
    else:
        indx = np.where(x >= np.array(starts))[0][-1]
        if x < list1[indx][1]: 
            if indx==0: return [-42, indx] # x in list1[0]
            else: return [indx-1, indx] # x in list1[indx] for indx>0
        else: return [indx, -42] # x follows list1[indx] but not in any interval

def excludeIntervals(list1, list2):
    """
    # exclude intervals in list2 from list1.
    assume intervals do not overlap
    e.g. list1 = [(0.1,10),(15.5,20)]; list2 = [(2,3.3),(5,6)]
    res = [(0.1, 2), (3.3, 5), (6, 10), (15.5, 20)]
    for 1 interval in list2: 11 possible situations regarding where its start and end points lie wrt intervals in list1. 3 of them do not require action.  
    another maybe more elegant way see Hodel's answer in stackoverflow 76362664 
    """
    list1 = [list(interval) for interval in list1]
    list2 = [list(interval) for interval in list2]
    list1.sort()
    list2.sort()
    res = list1
    for interval in list2:
        ind_start_before, ind_start = locateInIntervals(res, interval[0])
        ind_end_before, ind_end = locateInIntervals(res, interval[1])
        if ((ind_start_before < 0) & (ind_start < 0) & (ind_end >= 0)):
            res[ind_end] = [interval[1], res[ind_end][1]]
            for i in range(ind_end): del res[i]
        elif ((ind_start_before < 0) & (ind_start < 0) & (ind_end_before >= 0) & (ind_end < 0)):               
            for i in range(ind_end_before+1): del res[i]
        elif ((ind_start >= 0) & (ind_end >= 0)):
            if ind_start==ind_end:
                res.insert(ind_start+1, [interval[1], res[ind_end][1]])
                res[ind_start] = [res[ind_start][0], interval[0]]
            else:
                res[ind_start] = [res[ind_start][0], interval[0]]
                res[ind_end] = [interval[1], res[ind_end][1]]
        elif ((ind_start >= 0) & (ind_end_before >= 0) & (ind_end < 0)): 
            res[ind_start] = [res[ind_start][0], interval[0]]
            for i in range(ind_start+1, ind_end_before+1): del res[i]
        elif ((ind_start_before >= 0) & (ind_start < 0) & (ind_end >= 0)):
            res[ind_end] = [interval[1], res[ind_end][1]]
            for i in range(ind_start_before+1, ind_end): del res[i]
        elif ((ind_start_before >= 0) & (ind_start < 0) & (ind_end < 0)):
            for i in range(ind_start_before+1, ind_end_before+1): del res[i]
    return res



