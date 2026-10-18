"""Load data of the German distribution grid operators (§ 23c Abs. 3 EnWG).

Every operator publishes the same kinds of quarter-hourly series, each in
its own files: see docs/plans/german-grid-operators.md for the plan and
docs/plans/vnb-survey.md for what the files look like.

    timing.py   where each row of a file is in time: from its position,
                with its labels only checked
    cells.py    text cells into numbers, encodings, voltage levels
"""
