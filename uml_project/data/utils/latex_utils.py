# ------------------------ # Utility: Save TeX file from raw text # ------------------------
import re
from uml_project.data.constants import *


def save_tex_file(raw_text: str, base_name: str = "paper", strict_unescape: bool = False):
    """
    Save a raw LaTeX text string as a .tex file in the scratch directory,
    ensuring that literal '\\n' characters are treated as newlines **without**
    corrupting LaTeX control sequences like ``\newcommand``/``\newtheorem``.

    Parameters
    ----------
    raw_text : str
        The LaTeX text content, possibly containing literal '\\n' sequences.
    base_name : str
        The base name for the file (without extension). Defaults to 'paper'.
    strict_unescape : bool
        If True, aggressively replace *all* literal "\\n" with real newlines.
        Default False (safe mode): only unescape when "\\n" is **not** starting
        a LaTeX control word (i.e., not followed by a letter).

    Returns
    -------
    str
        The full path to the saved .tex file.

    Notes
    -----
    - The naive replacement of all "\\n" → "\\n" breaks LaTeX commands like
      \newcommand or \newtheorem, which must remain as literal backslash + n.
    - This function only replaces "\\n" when it is *not* followed by a letter,
      leaving LaTeX commands untouched. See comments below for details.
    - If strict_unescape=True, all "\\n" are replaced, for trusted JSON input.
    """
    import os
    from datetime import datetime

    scratch_dir = SCRATCH_DIR
    os.makedirs(scratch_dir, exist_ok=True)

    # Conservatively convert escaped newlines.
    # The naive replacement of all "\\n" -> "\n" breaks LaTeX commands like \newcommand / \newtheorem
    # (which literally begin with the two characters backslash + 'n').
    #
    # Safe rule: only convert "\\n" when it is **not** followed by a letter.
    # This handles JSON-style line breaks (often followed by '\\', digits, space, etc.)
    # while preserving control words like "\\newcommand".
    #
    # Do not touch "\\\n" (double backslash + n) which is a LaTeX linebreak.
    if "\\n" in raw_text:
        if strict_unescape:
            # Strict: replace all (legacy)
            text_content = raw_text.replace("\\r\\n", "\n").replace("\\n", "\n")
        else:
            text_content = raw_text
            # Normalize escaped CRLF first
            text_content = text_content.replace("\\r\\n", "\n")
            # Replace \n not followed by a letter (negative lookahead)
            # This preserves e.g. \newcommand, \noindent, \newtheorem, etc.
            text_content = re.sub(r"\\n(?![A-Za-z])", "\n", text_content)
    else:
        text_content = raw_text

    # Create timestamped filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.tex"
    file_path = os.path.join(scratch_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    return text_content, file_path
