from __future__ import absolute_import

class Parameters(object):
    """
    Takes in the parameters and freeindex. Has a bunch of utility
    functions to transform parameters in various ways.
    """

    def __init__(self, parameters, values):
        self.parameters = parameters # the output from problem.parameters()
        self.values     = values
        self.constants  = [param[0] for param in parameters]
        self.labels     = [param[1] for param in parameters]

        # Set all the constants to their initial values
        for (j, label) in enumerate(self.labels):
            const = self.constants[j]
            val   = values[label]
            const.assign(val[0])

    def update_constants(self, values):
        for (val, const) in zip(values, self.constants):
            const.assign(val)

    def floats(self, freeindex=None, value=None):
        out = list(map(float, self.constants))
        if value is not None and freeindex is not None:
            out[freeindex] = value
        return tuple(out)

    def next(self, oldparams, freeindex):
        """
        This is probably unnecessarily slow. It will index into
        values each time it is called to find the next parameter
        value. We could think about using a better data structure
        for values.
        """
        current_value = oldparams[freeindex]
        label = self.labels[freeindex]
        current_index = self.values[label].index(current_value)

        if current_index == len(self.values[label]) - 1:
            # We've reached the last value, no more continuation to do.
            return None
        next_value = self.values[label][current_index + 1]

        newparams = list(oldparams)
        newparams[freeindex] = next_value
        return tuple(newparams)

    def previous(self, oldparams, freeindex):
        current_value = oldparams[freeindex]
        label = self.labels[freeindex]
        current_index = self.values[label].index(current_value)

        if current_index == 0:
            # We've reached the first value, no more continuation to do.
            return None
        prev_value = self.values[label][current_index - 1]

        newparams = list(oldparams)
        newparams[freeindex] = prev_value
        return tuple(newparams)

    def update_from_string(s):
        subs = s.split('@')
        assert len(subs) == len(self.parameters)

        for (sub, const) in zip(subs, self.constants):
            val = float(sub.split('=')[1])
            const.assign(val)

def parameters_to_string(parameters, values):
    s = ""
    for (param, value) in zip(parameters, values):
        s += "%s=%.15e@" % (param[1], value)
    return s[:-1]

