#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
from jsoncomment import JsonComment

if __name__ == "__main__":
    parserJSON = JsonComment(json)
    with open(sys.argv[1], "r") as f:
        s = "".join(f.readlines())

    data = parserJSON.loads(s.replace('\t', '').replace("\n","").replace("\M","").replace(",  ]","]"))

    with open(sys.argv[2],"w") as f:
        json.dump(data,f)
