#Programming for the Puzzled -- Srini Devadas
#You Will All Conform
#Input is a vector of F's and B's, in terms of forwards and backwards caps
#Output is a set of commands (printed out) to get either all F's or all B's
#Fewest commands are the goal

caps = ['F', 'F', 'B', 'B', 'B', 'F', 'B',
        'B', 'B', 'F', 'F', 'B', 'F' ]

cap2 = ['F', 'F', 'B', 'B', 'B', 'F', 'B',
        'B', 'B', 'F', 'F', 'F', 'F' ]

def pleaseConform(caps):
    #Initialization
    start = None
    forward = 0
    backward = 0
    intervals = []

    #Determine intervals where caps are on in the same direction
    for i in range(len(caps)):

        if caps[i] == 'H':
            if start is not None:
                intervals.append((start, i - 1, caps[start]))

                if caps[start] == 'F':
                    forward += 1
                else:
                    backward += 1

                start = None

        elif start is None:
            start = i

        elif caps[start] != caps[i]:
            intervals.append((start, i - 1, caps[start]))

            if caps[start] == 'F':
                forward += 1
            else:
                backward += 1

            start = i

    #Add the last interval
    if start is not None:
        intervals.append((start, len(caps) - 1, caps[start]))

        if caps[start] == 'F':
            forward += 1
        else:
            backward += 1

    ## print(intervals)
    ## print(forward, backward)

    if forward < backward:
        flip = 'F'
    else:
        flip = 'B'

    for t in intervals:
        if t[2] == flip:
            if t[0] == t[1]:
                print('Person in position', t[0], 'flip your cap!')
            else:
                print('People in positions', t[0],
                      'through', t[1], 'flip your caps!')


pleaseConform(caps)
##pleaseConform(cap2)