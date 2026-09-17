# -*- coding: utf-8 -*-
import docx
from docx.shared import Cm

SRC = r"C:\Users\Akshat Jha\Downloads\write-up-longer.docx"

doc = docx.Document(SRC)

resized = 0
for shape in doc.inline_shapes:
    shape.width = Cm(2.7)
    shape.height = Cm(2.7)
    resized += 1

doc.save(SRC)
print("resized:", resized)
