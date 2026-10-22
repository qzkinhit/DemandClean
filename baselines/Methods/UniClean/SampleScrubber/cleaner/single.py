import re
import time
from collections import Counter
from datetime import datetime

import numpy as np
from dateparser.date import DateDataParser

from SampleScrubber.cleaner_model import CleanerModel, ParametrizedLanguage, Uniop
from SampleScrubber.util.format import formatString
from collections import Counter
import pandas as pd

"""
    Constraints applied row by row to a single attribute, usually in the precleaning stage.
"""


class Single(CleanerModel):
    """
        Predicate constraint: the single-attribute form of DataModel, the parent class of the other constraints.
    """

    def __init__(self, domain, check, msg, format, none_penalty=0.01, condition_func=None, ExtDict=None, edit_rule=None,
                 invalidDict=None):
        """The constructor takes an attribute and a function that maps attribute values to true or false.

        attr -- attribute the predicate applies to
        check -- a function func(attr) -> {true, false}
        none_penalty -- penalty of None
        """
        self.msg = msg
        self.domain = domain
        self.check = check
        self.format = format
        self.none_penalty = none_penalty
        super(Single, self).__init__(domain=self.domain, source=self.domain, target=self.domain,
                                     msg=self.msg)  # Call the constructor of the parent class
        if condition_func is not None:
            # condition_func maps the tuple of source values to true or false
            self.fixValueRules['condition_func'] = condition_func
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

    def _quality(self, df):
        N = df.shape[0]
        qfn_a = np.ones((N,))
        for i in range(N):
            val = df[self.domain].iloc[i]  # Attribute value of row i
            if val == None:
                qfn_a[i] = self.none_penalty  # Penalty of None
            elif self.check(val):  # Does the attribute satisfy the rule
                qfn_a[i] = 0
        return qfn_a  # Quality score

    def preProcess(self):
        raise NotImplementedError("precleaning mode of this single-column operator not implemented")


class Date(Single):
    """Date: a datetime string field must have a given format.
    """

    def __init__(self, domain, format, name, valid_date_pattern=None, currenFormat=None,condition_func=None, ExtDict=None, edit_rule=None,
                 invalidDict=None):
        """
        attr -- attribute name
        pattern -- a standard Python strftime string
        """
        self.name = name
        self.format = format
        self.domain = domain
        self.msg = '[Date:(domain=%s, format = %s)]' % (self.domain, self.format)
        self.valid_date_pattern = valid_date_pattern
        self.currenFormat=currenFormat
        if condition_func is not None:
            # condition_func maps the tuple of source values to true or false
            self.fixValueRules['condition_func'] = condition_func
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

        def timePatternCheck(x, p,v):
            # Empty or NaN
            if x == None or x != x or len(x.strip()) == 0:
                return False

            try:
                if re.match(valid_date_pattern, x) or not v:
                    # True if the string parses as a time, False otherwise
                    t = time.strptime(x, p)
                    return True
            except ValueError:
                return False

        super(Date, self).__init__(domain, lambda x, p=format,v=valid_date_pattern: timePatternCheck(x, p,v), self.msg, self.format)

    def __str__(self):
        return self.msg

    from datetime import datetime
    import re
    from dateparser.date import DateDataParser

    def preProcess(self):
        dateInstance = self

        class PreLanguage(ParametrizedLanguage):
            paramDescriptor = {
                'domain': ParametrizedLanguage.DOMAIN,
                'format': ParametrizedLanguage.FORMAT,
                'opinformation': ParametrizedLanguage.OPINFORMATION,
            }

            def __init__(self, dateInstance):
                """
                Datetime conversion operator.

                Parameters:
                column: str, datetime column to convert
                format: str, datetime format string
                opinformation: str, name of the cleaning signal
                """
                # Parameters of dateInstance
                domain = dateInstance.domain
                format = dateInstance.format
                valid_date_pattern = dateInstance.valid_date_pattern
                currenFormat = dateInstance.currenFormat
                opinformation = str(dateInstance)
                parser = DateDataParser(languages=['en'])

                def fn(df, column=domain, format=format, parser=parser, valid_date_pattern=valid_date_pattern,
                       currenFormat=currenFormat):
                    # DataParser
                    if parser is None:
                        parser = DateDataParser(languages=['en'])

                    column_name = column
                    editRuleList = []  # Edit rules

                    # Does the format contain a year but no day
                    contains_year_in_double_obj = format.count('%') == 2 and ("y" in format or "Y" in format)
                    # With a year, extract the separator
                    separator = None
                    if contains_year_in_double_obj:
                        separator_match = re.search(r'[^%\w]', re.sub(r'%\w', '', format))
                        separator = separator_match.group(0) if separator_match else None

                    # Regular expression of the year
                    year_pattern = r'^\d{2}$|^\d{4}$'

                    for i in df.index:
                        cell_value = str(df.at[i, column_name])  # Value as a string

                        # Match valid_date_pattern first, if set
                        if valid_date_pattern and not re.match(valid_date_pattern, cell_value):
                            continue

                        if cell_value:
                            try:
                                # Split the string and detect the year
                                year_str = None
                                # Which side is the year
                                if separator and contains_year_in_double_obj:
                                    parts = cell_value.split(separator)
                                    left_part, right_part = parts[0], parts[1]
                                    # Can the left or the right part be a year
                                    if re.match(year_pattern, left_part):
                                        year_str = left_part
                                    elif re.match(year_pattern, right_part):
                                        year_str = right_part

                                # Parse with currenFormat or with DateDataParser
                                if currenFormat:
                                    try:
                                        # First, parse directly
                                        parsed_date = datetime.strptime(cell_value, currenFormat)
                                    except ValueError:
                                        # On failure, prepend 0 and parse again
                                        try:
                                            cell_value_edit = "0" + cell_value
                                            parsed_date = datetime.strptime(cell_value_edit, currenFormat)
                                        except ValueError:
                                            parsed_date = None
                                else:
                                    parsed_data = parser.get_date_data(cell_value)
                                    parsed_date = parsed_data['date_obj']

                                # Parsed and a year found
                                if parsed_date and year_str and contains_year_in_double_obj:
                                    # Two-digit years become four-digit years after 2000
                                    parsed_year = int(year_str) if len(year_str) == 4 else int("20" + year_str)
                                    if parsed_date.year != parsed_year:
                                        parsed_date = parsed_date.replace(year=parsed_year)

                                # Format the date in the target format
                                formatted_date = parsed_date.strftime(format)

                                # Does the value change
                                if formatted_date != cell_value:
                                    # Repair operation
                                    predicate = ([column_name], {(cell_value,)}, {i})
                                    opinformation = f"Date({column_name},{format})"
                                    editRule = Uniop(
                                        domain=column_name,
                                        repairvalue=formatted_date,
                                        predicate=predicate,
                                        opinformation=opinformation
                                    )
                                    editRuleList.append(editRule)

                                # Update the DataFrame
                                df.at[i, column_name] = formatted_date
                            except:
                                df.at[i, column_name] = '__NULL__'

                    return df, editRuleList  # Updated DataFrame and edit rules

                # Class name and provenance
                self.name = 'df = dateparse(df,' + formatString(domain) + ',' + formatString(format) + str(
                    opinformation) + ')'
                self.provenance = [self]

                # Call the constructor of the parent class
                super(PreLanguage, self).__init__(fn, ['domain', 'format', 'opinformation'])

        return PreLanguage(dateInstance)
class DisguisedMissHandler(Single):
    """DisguisedMissHandler: handles disguised missing values of a field.
    """

    def __init__(self, domain, format=None, name="MissingValueHandler", missing_values=None):
        """
        domain -- attribute checked for missing values
        format -- replacement value, the most frequent value of the column if not given
        missing_values -- strings that denote missing values (for example ["", None, "empty", "null"])
        """
        self.name = name
        self.format = format
        self.domain = domain
        self.msg = '[MissingValueHandler: (domain=%s, format=%s, missing_values=%s)]' % (
            self.domain, self.format, missing_values)
        self.missing_values = missing_values if missing_values is not None else ["", None, "NaN", "null", "empty"]

        def missCheck(x, missing_values):
            # Empty, NaN or a missing-value token
            if x is None or x != x or x.strip() == "" or x in missing_values:
                return False
            return True
        # Initialize with missCheck
        super(DisguisedMissHandler, self).__init__(domain, lambda x, m=missing_values: missCheck(x, m), self.msg,
                                                  format=self.format)

    def __str__(self):
        return self.msg

    def handle_missing(self, df):
        """Handle the missing values."""
        column_name = self.domain
        editRuleList = []  # Edit rules

        # Without format, the most frequent valid value of the column
        replacement_value = self.format
        if replacement_value is None:
            # Drop empty and disguised missing values
            valid_data = df[column_name][~df[column_name].isin(self.missing_values)].dropna()
            most_common_value = valid_data.mode()

            if not most_common_value.empty:
                replacement_value = most_common_value[0]
            else:
                replacement_value = '__NULL__'  # Default placeholder if no value is valid

        # Replace value by value
        for i in df.index:
            cell_value = df.at[i, column_name]
            # Is it missing
            if pd.isna(cell_value) or cell_value in self.missing_values and replacement_value != '__NULL__':
                # Replacement operation
                predicate = ([column_name], {(cell_value,)}, {i})
                opinformation = f"MissingValueHandler({column_name}, {replacement_value})"
                editRule = Uniop(
                    domain=column_name,
                    repairvalue=replacement_value,
                    predicate=predicate,
                    opinformation=opinformation
                )
                editRuleList.append(editRule)

                # Update the DataFrame
                df.at[i, column_name] = replacement_value

        return df, editRuleList  # Updated DataFrame and edit rules

    def preProcess(self):
        missingValueInstance = self

        class PreLanguage(ParametrizedLanguage):
            paramDescriptor = {'domain': ParametrizedLanguage.DOMAIN,
                               'format': ParametrizedLanguage.FORMAT,
                               'opinformation': ParametrizedLanguage.OPINFORMATION,
                               }

            def __init__(self, missingValueInstance):
                """
                Missing-value operator.

                Parameters:
                domain: str, column with missing values
                format: str, replacement value, the most frequent value of the column if not given
                opinformation: str, name of the cleaning signal
                """
                domain = missingValueInstance.domain
                format = missingValueInstance.format
                opinformation = str(missingValueInstance)
                missing_values = missingValueInstance.missing_values

                def fn(df, column=domain, format=format, missing_values=missing_values):
                    column_name = column
                    editRuleList = []  # Edit rules

                    # Without format, the most frequent valid value of the column
                    replacement_value = format
                    if replacement_value is None:
                        # Drop empty and disguised missing values
                        valid_data = df[column_name][~df[column_name].isin(missing_values)].dropna()
                        most_common_value = valid_data.mode()

                        if not most_common_value.empty:
                            replacement_value = most_common_value[0]
                        else:
                            replacement_value = '__NULL__'  # Default placeholder if no value is valid

                    # Replace value by value
                    for i in df.index:
                        cell_value = df.at[i, column_name]
                        # Is it missing
                        if pd.isna(cell_value) or cell_value in missing_values:
                            # Replacement operation
                            predicate = ([column_name], {("__NULL__",)}, {i})
                            opinformation = f"MissingValueHandler({column_name}, {replacement_value})"
                            editRule = Uniop(
                                domain=column_name,
                                repairvalue=replacement_value,
                                predicate=predicate,
                                opinformation=opinformation
                            )
                            editRuleList.append(editRule)

                            # Update the DataFrame
                            df.at[i, column_name] = replacement_value

                    return df, editRuleList  # Updated DataFrame and edit rules

                self.name = 'df = MissingValueHandler(df,' + formatString(domain) + ',' + formatString(format) + str(
                    opinformation) + ')'
                self.provenance = [self]

                super(PreLanguage, self).__init__(fn, ['domain', 'format', 'opinformation'])

        return PreLanguage(missingValueInstance)


class Number(Single):
    """Number: the attribute must be numeric within a range."""

    def __init__(self, domain, nrange=None, name=None, ExtDict=None, edit_rule=None, invalidDict=None):
        """
        Parameters:
        domain: str, column name
        nrange: list, allowed range, [-inf, inf] by default
        name: str, name
        """
        if nrange is None:
            nrange = [-np.inf, np.inf]
        self.name = name
        self.attr = domain
        self.format = nrange
        self.msg = '[Number:(domain=%s, format=%s)]' % (self.attr, str(self.format))

        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

        def floatPatternCheck(x):
            # Empty or a float
            if x is None or isinstance(x, float):
                return True
            else:
                return False

        super(Number, self).__init__(domain, lambda x: floatPatternCheck(x), self.msg, self.format)

    def __str__(self):
        return self.msg

    def preProcess(self):
        floatInstance = self

        class PreLanguage(ParametrizedLanguage):
            paramDescriptor = {
                'domain': ParametrizedLanguage.DOMAIN,
                'format': ParametrizedLanguage.FORMAT,
                'opinformation': ParametrizedLanguage.OPINFORMATION,
            }

            def __init__(self, floatInstance):
                """
                Float conversion operator.

                Parameters:
                domain: str, column to convert
                format: list, range of the floats
                opinformation: str, name of the cleaning signal
                """
                domain = floatInstance.attr
                format = floatInstance.format
                opinformation = f"Number({formatString(domain)}, range={format})"

                def fn(df, domain=domain, r=format):
                    editRuleList = []  # Generated Uniop objects

                    def extract_first_number(value):
                        """First number of a string, non-numeric characters removed."""
                        # Remove the letters
                        value_without_letters = re.sub(r'[a-zA-Z]', '', value)
                        # First number
                        match = re.search(r'[-+]?\d*\.?\d+', value_without_letters)
                        return float(match.group(0)) if match else None
                    def __internal(row):
                        try:
                            original_value = row[domain]
                            if isinstance(original_value, str):
                                # A purely numeric string converts with float() and needs no edit rule
                                try:
                                    value = float(original_value)
                                    if r[0] <= value <= r[1]:
                                        return value
                                    else:
                                        return '__NULL__'
                                except (ValueError, TypeError):
                                    # Other strings (letters or symbols): extract the number and record the edit
                                    value = extract_first_number(original_value)
                                    if value is not None and r[0] <= value <= r[1]:
                                        predicate = ([domain], {(original_value,)}, {row.name})
                                        editRule = Uniop(
                                            domain=domain,
                                            repairvalue=value,
                                            predicate=predicate,
                                            opinformation=opinformation
                                        )
                                        editRuleList.append(editRule)
                                        return value
                                    else:
                                        return '__NULL__'
                            else:
                                # Already numeric (int or float), no edit rule
                                value = float(original_value)
                                if value is not None and r[0] <= value <= r[1]:
                                    return value
                                else:
                                    return '__NULL__'
                        except:
                            return '__NULL__'

                    # Apply the repairs
                    df[domain] = df.apply(lambda row: __internal(row), axis=1)
                    return df, editRuleList  # DataFrame and edit rules

                self.name = 'df = numparse(df,' + formatString(domain) + ',' + str(opinformation) + ')'
                self.provenance = [self]

                super(PreLanguage, self).__init__(fn, ['domain', 'format', 'opinformation'])

        return PreLanguage(floatInstance)


class Pattern(Single):
    """Pattern: a string field must match a regular expression.
    """

    def __init__(self, domain, format, name, condition_func=None, ExtDict=None, edit_rule=None, invalidDict=None):
        """attr -- attribute name
        pattern -- a standard Python regular expression
        """
        self.name = name
        self.format = format
        self.domain = domain
        self.msg = '[Pattern:(domain= %s, format = %s)]' % (self.domain, str(self.format))
        if condition_func is not None:
            # condition_func maps the tuple of source values to true or false
            self.fixValueRules['condition_func'] = condition_func
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

        def patternCheck(x, p):
            # Not empty or NaN and matches the regular expression
            if x is None or x == '' or str(x).strip() == '':
                return False

            try:
                result = re.match(p, x)
                return bool(result)
            except:
                return False

        super(Pattern, self).__init__(domain, lambda x, p=format: patternCheck(x, p), self.msg, self.format)

    def __str__(self):
        # return 'attribute %s has type Pattern, rule %s' % (self.domain, str(self.pattern))
        return self.msg

    def preProcess(self):
        patternInstance = self

        class PreLanguage(ParametrizedLanguage):
            paramDescriptor = {'domain': ParametrizedLanguage.DOMAIN,
                               'format': ParametrizedLanguage.FORMAT,
                               'opinformation': ParametrizedLanguage.OPINFORMATION,
                               }

            def __init__(self, patternInstance):
                """
                Regular-expression conversion operator.

                Parameters:
                domain: str, column to convert
                format: str, pattern
                opinformation: str, name of the cleaning signal
                """
                domain = patternInstance.domain
                format = patternInstance.format
                opinformation = str(patternInstance)

                def extract_first_last_match(pattern):
                    # First part that matches the pattern
                    start_match = re.match(r"^\^?(\[[^\]]+\]|[A-Za-z0-9]+|\([^)]+\))", pattern)
                    first_part = start_match.group(1) if start_match else None

                    # Last part that matches the pattern
                    end_match = re.search(r"(\([^)]+\)|\[[^\]]+\]|[A-Za-z0-9]+)\$?$", pattern)
                    last_part = end_match.group(1) if end_match else None

                    return first_part, last_part

                def trim_to_nearest_match(string, pattern):
                    """
                    Tries to match the pattern directly to the string. If it matches, return the match.
                    If not, iteratively expand from the first and last matched parts of the pattern within the string.
                    """
                    # Attempt a full match first
                    full_match = re.match(pattern, string)
                    if full_match:
                        return full_match.group()  # Return if full match is found

                    # Extract first and last identifiable groups from the pattern
                    first_part, last_part = extract_first_last_match(pattern)

                    # Build a new regex to capture nearest range in the string
                    if first_part and last_part:
                        regex = re.compile(rf"{first_part}.*?{last_part}", re.IGNORECASE)
                        partial_match = regex.search(string)
                        if partial_match:
                            return partial_match.group()  # Return closest match within the bounds of first and last

                    return string  # Return original string if no match found

                def fn(df, domain=domain, format=format):
                    editRuleList = []  # Generated Uniop objects
                    column_name = domain

                    for i in range(len(df)):
                        cell_value = df.at[i, column_name]
                        if bool(cell_value):
                            try:
                                # Step 1: search for a matching substring
                                match = re.search(format, str(cell_value))
                                if match:
                                    matched_value = match.group(0)
                                else:
                                    # Step 2: trim with trim_to_nearest_match
                                    matched_value = trim_to_nearest_match(str(cell_value), format)

                                # Step 3: a trimmed value that still does not match becomes "__NULL__"
                                if not re.match(format, matched_value):
                                    matched_value = "__NULL__"

                                # DataFrame update
                                if matched_value != "__NULL__" and str(cell_value) != matched_value:
                                    # Predicate in the given format
                                    predicate = ([column_name], {(cell_value,)}, {i})

                                    # opinformation field
                                    opinformation = f"Pattern({formatString(domain)},{formatString(format)})"

                                    # New Uniop object added to editRuleList
                                    uniop = Uniop(domain=column_name,
                                                  repairvalue=matched_value,
                                                  predicate=predicate,
                                                  opinformation=opinformation)
                                    editRuleList.append(uniop)
                                    self.set_reward(self.reward + 1)

                                # Update the DataFrame
                                df.at[i, column_name] = matched_value

                            except Exception as e:
                                # On an exception the value becomes "__NULL__"
                                df.at[i, column_name] = "__NULL__"

                    return df, editRuleList  # DataFrame and Uniop objects

                self.name = 'df = pattern(df,' + formatString(domain) + ',' + formatString(format) + str(
                    opinformation) + ')'
                self.provenance = [self]

                super(PreLanguage, self).__init__(fn, ['domain', 'format', 'opinformation'])

        return PreLanguage(patternInstance)


class DictValue(Single):
    """
    DictValue: the values of an attribute must come from a dictionary
    """

    def __init__(self, domain, ExtDict, sim=0.1, condition_func=None, edit_rule=None, invalidDict=None):
        """ Constructor of DictValue.

        Positional arguments:
        attr -- an attribute name
        codebook -- dictionary of the allowed domain
        """
        self.attr = domain
        self.sim = sim
        self.ExtDict = ExtDict
        super(DictValue, self).__init__(domain, lambda x, ExtDict=self.ExtDict: x in ExtDict)
        if condition_func is not None:
            # condition_func maps the tuple of source values to true or false
            self.fixValueRules['condition_func'] = condition_func
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

    def generateExtDict(df, col, size=100):
        """
        Helper that selects the k most frequent values of the attribute
        """

        """Count the values of col, a dictionary (value, count)."""
        s = Counter([v for v in df[col].values if v == str(v)])

        """Sort by count in descending order, l holds (count, value)."""
        l = [i for i in sorted([(s[k], k) for k in s], reverse=True)]
        # print(l)

        """Slice of length size, holding values."""
        ExtDict = set([occ[1] for occ in l][0:size])

        return ExtDict

    def generateCorrelationExtDict(d, attr, labels, size=100):
        values = set([v for v in d[attr].values if v == str(v)])

        ranked_tokens = []

        for v in values:
            """Correlation coefficient."""
            corr = np.abs(np.corrcoef(d[attr] == v, labels)[0, 1])
            """Add to ranked_tokens."""
            ranked_tokens.append((corr, v))
        """Sort in descending order."""
        ranked_tokens.sort(reverse=True)

        return set([v[1] for v in ranked_tokens[:size]])


class Outlier(Single):
    """Outlier: detects and cleans outliers of an attribute, from a user-given set and built-in rules.
    """

    def __init__(self, domain, format=None, name="Outlier", condition_func=None, ExtDict=None, edit_rule=None,
                 invalidDict=None):
        """
        attr -- attribute name
        format -- set of outlier values
        """
        self.name = name
        self.domain = domain
        self.format = format or set()
        self.msg = '[Outlier:(domain=%s, outliers= %s)]' % (self.domain, self.format)
        if condition_func is not None:
            self.fixValueRules['condition_func'] = condition_func
        if ExtDict is not None:
            self.fixValueRules['ExtDict'] = ExtDict
        if invalidDict is not None:
            self.fixValueRules['invalidDict'] = invalidDict
        if edit_rule is not None:
            self.fixValueRules['edit_rule'] = edit_rule

        def outlierCheck(x, outliers):
            # Empty, NaN or in the outlier set
            if x is None or x != x or x.strip() == "" or any(outlier in x for outlier in outliers):
                return False
            return True

        # Pass format to super
        super(Outlier, self).__init__(domain, lambda x, o=self.format: outlierCheck(x, o), self.msg, format=self.format)

    def __str__(self):
        return self.msg

    def preProcess(self):
        outlierInstance = self

        class PreLanguage(ParametrizedLanguage):
            paramDescriptor = {'domain': ParametrizedLanguage.DOMAIN,
                               'format': ParametrizedLanguage.FORMAT,
                               'opinformation': ParametrizedLanguage.OPINFORMATION,
                               }

            def __init__(self, outlierInstance):
                """
                Outlier cleaning operator.

                Parameters:
                domain: str, column to clean
                format: set, outlier values
                opinformation: str, name of the cleaning signal
                """
                domain = outlierInstance.domain
                outliers = outlierInstance.format
                opinformation = str(outlierInstance)

                def fn(df, column=domain, outliers=outliers):
                    column_name = column
                    editRuleList = []  # Generated Uniop objects

                    def remove_outliers(value):
                        # Remove the given characters
                        for outlier in outliers:
                            value = value.replace(outlier, "")
                        # Remove repeated double and single quotes
                        value = re.sub(r"'{2}", "'", value)  # Two single quotes become one
                        value = re.sub(r'"{2}', '"', value)  # Two double quotes become one
                        # Remove non-ASCII characters
                        value = re.sub(r'[^\x00-\x7F]+', '', value)
                        # Collapse whitespace
                        return value.strip()

                    for i in df.index:
                        cell_value = str(df.at[i, column_name]).strip()
                        original_value = cell_value
                        # Remove outliers and repeated quotes
                        cell_value = remove_outliers(cell_value)

                        # A changed value gets a repair operation
                        if cell_value != original_value:
                            predicate = ([column_name], {(original_value,)}, {i})
                            opinformation = f"Outlier({formatString(column)}, Outliers={outliers})"
                            editRule = Uniop(
                                domain=column_name,
                                repairvalue=cell_value,
                                predicate=predicate,
                                opinformation=opinformation
                            )
                            editRuleList.append(editRule)
                            df.at[i, column_name] = cell_value  # Update the DataFrame values

                    return df, editRuleList  # DataFrame and edit rules

                self.name = 'df = outlier_clean(df,' + formatString(domain) + ',' + str(opinformation) + ')'
                self.provenance = [self]

                super(PreLanguage, self).__init__(fn, ['domain', 'format', 'opinformation'])

        return PreLanguage(outlierInstance)
