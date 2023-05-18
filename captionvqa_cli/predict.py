#!/usr/bin/env python3 -u


import sys

from captionvqa_cli.run import run


def predict(opts=None):
    if opts is None:
        sys.argv.extend(["evaluation.predict=true"])
    else:
        opts.extend(["evaluation.predict=true"])

    run(predict=True)


if __name__ == "__main__":
    predict()
