#!/usr/bin/env python
"""Public entry point for Figures 4 and 5 of the paper.

    python figures/build_figure_data.py
    python figures/gen_figures.py --out figures

This adapter uses the same plotting implementation as make_figures.py and the final
inputs in figures/data. It needs no research workspace, private path, or run search.
"""
from make_figures import main


if __name__ == '__main__':
    main()
