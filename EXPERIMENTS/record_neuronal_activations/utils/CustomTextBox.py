import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.path import Path
from matplotlib.patches import BoxStyle


class ExtendedTextBox(BoxStyle._Base):
    """
    An Extended Text Box that expands to the axes limits
                        if set in the middle of the axes
    """

    def __init__(self, pad=0.3, width=500.):
        """
        width:
            width of the textbox.
            Use `ax.get_window_extent().width`
                   to get the width of the axes.
        pad:
            amount of padding (in vertical direction only)
        """
        self.width=width
        self.pad = pad
        super(ExtendedTextBox, self).__init__()

    def transmute(self, x0, y0, width, height, mutation_size):
        """
        x0 and y0 are the lower left corner of original text box
        They are set automatically by matplotlib
        """
        # padding
        pad = mutation_size * self.pad

        # we add the padding only to the box height
        height = height + 2.*pad
        # boundary of the padded box
        y0 = y0 - pad
        y1 = y0 + height
        _x0 = x0
        x0 = _x0 +width /2. - self.width/2.
        x1 = _x0 +width /2. + self.width/2.

        cp = [(x0, y0),
              (x1, y0), (x1, y1), (x0, y1),
              (x0, y0)]

        com = [Path.MOVETO,
               Path.LINETO, Path.LINETO, Path.LINETO,
               Path.CLOSEPOLY]

        path = Path(cp, com)

        return path
