# -*- coding: utf-8 -*-
import re

def isHangeul(text):

    encText = text

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0


