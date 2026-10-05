"""Document forensics (Version 3): signs that a file was altered.

    model.py     what an indicator is and how indicators add up to a score
    pdf.py       what a PDF shows to a reader, and what it hides
    docx.py      hidden text, tracked changes and file history of Word documents
    text.py      fields that are stated twice with different values
    analyze.py   run the right checks for one uploaded file
    store.py     where the results are kept

Every result is a risk indicator, never proof. A re-saved PDF, a corporate
template with two fonts or an OCR layer are all innocent, and the checks are
written so that those alone do not raise the score. What the module can say
is "a person should look at this file before relying on it".

Not built, on purpose: pixel-level image forensics (compression analysis,
copy-move detection). Without a trained and validated model such scores look
precise and are not, which is worse than not having them.
"""
