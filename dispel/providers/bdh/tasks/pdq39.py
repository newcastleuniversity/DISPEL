"""Functionalities to process pdq39 data."""

from typing import List

import pandas as pd

from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import ExtractStep
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.registry import process_factory

TASK_NAME = AV("Parkinson Disease Questionnaire", "pdq39")

DOMAINS = {
    "mobility": range(0, 10),
    "activities_of_daily_living ": range(10, 16),
    "emotional_well_being": range(16, 22),
    "stigma": range(22, 26),
    "social_support": range(26, 29),
    "cognition": range(29, 33),
    "communication": range(33, 36),
    "bodily_discomfort": range(36, 39),
}
r"""The eight functional domains that were evaluated through the survey."""


class TransformAnswerInScore(TransformStep):
    """Transform answers in score between 0 and 4."""

    data_set_ids = "userInput"
    new_data_set_id = "score_per_answer_w_category"

    @staticmethod
    @transformation
    def transform_answer_in_score(data: pd.DataFrame) -> pd.DataFrame:
        """Transform answer into a score between 0 and 4."""
        df = pd.DataFrame(
            {"question_number": data.displayedValue, "answer": data.answer}
        )
        # Create the score between 0 and 4
        df["score"] = data.answer - 1
        # Specific case for the 28th question with a possibility to answer
        # six different answers. The score of the first two being the same,
        # because (first answer is "I do not have a spouse or partner" and the
        # second answer is "Never") we consider they should have the same value
        # of 0 (best qol).
        df.loc[27, "score"] = max(0, df.loc[27, "score"] - 1)

        # Add the category
        df["domain"] = None

        for domain, _range in DOMAINS.items():
            df.loc[_range, "domain"] = domain

        return df

    definitions = [
        RawDataValueDefinition("question_number", "The question number."),
        RawDataValueDefinition(
            "answer", "The answer to the question with values between 1 and 6."
        ),
        RawDataValueDefinition(
            "score", "The score with values between 0 and 4 (lower the better)."
        ),
        RawDataValueDefinition(
            "domain", "The domain of quality of life being assessed by the question."
        ),
    ]


class GroupScorePerDomain(TransformStep):
    """Transform question scores per domain and creates domain scores."""

    data_set_ids = "score_per_answer_w_category"
    new_data_set_id = "domain_scores"

    @staticmethod
    @transformation
    def group_score_per_domain(data: pd.DataFrame) -> pd.DataFrame:
        """Group question scores per domain and creates domain scores."""
        group = data[["domain", "score"]].groupby("domain")
        res = group.count()
        res.rename(columns={"score": "count"}, inplace=True)
        res["domain_score"] = group.sum()
        res["domain_score_normalized"] = (res["domain_score"] / res["count"]) * 25
        return res

    definitions = [
        RawDataValueDefinition("count", "The number of questions in the domain."),
        RawDataValueDefinition(
            "domain_score",
            "The sum of the score (between 0 and 4) to each individual answer "
            "that belongs to the domain.",
        ),
        RawDataValueDefinition(
            "domain_score_normalized", "The domain score normalized between 0 and 100."
        ),
    ]


class ExtractDomainScore(ExtractStep):
    """Extract PDQ-39 domain score."""

    data_set_ids = "domain_scores"

    def __init__(self, domain: str, *args, **kwargs):
        self.domain = domain
        super().__init__(*args, **kwargs)

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Overwrite get_definition."""
        description = (
            f"PDQ domain score for {self.domain}. It is computed "
            "as the sum of the scores (ranging between 0 and 4) "
            f"for all the questions of the domain {self.domain}."
        )

        return MeasureValueDefinitionPrototype(
            measure_name=AV(f"{self.domain} score", f"{self.domain}_score"),
            data_type="float",
            validator=RangeValidator(lower_bound=0, upper_bound=100),
            description=description,
        ).create_definition(**kwargs)

    @transformation
    def read_pdq_domain_score(self, data: pd.DataFrame) -> float:
        """Read pdq normalized score corresponding to the domain."""
        return float(data.loc[self.domain, "domain_score_normalized"])


class ExtractTotalScore(ExtractStep):
    """Extract PDQ-39 total score."""

    data_set_ids = "domain_scores"

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("Total PDQ-39 score", "total_score"),
        data_type="float",
        validator=RangeValidator(lower_bound=0, upper_bound=100),
        description="The total PDQ-39 score is the mean of the eight domains "
        "scores (one for each domain evaluated) and has a value"
        "between 0 and 100.",
    )

    @staticmethod
    @transformation
    def read_pdq_total_score(data: pd.DataFrame) -> float:
        """Read PDQ39 total score."""
        return data.domain_score_normalized.mean()


STEPS: List[ProcessingStep] = [
    TransformAnswerInScore(),
    GroupScorePerDomain(),
    *[ExtractDomainScore(domain=domain) for domain in DOMAINS],
    ExtractTotalScore(),
]
STEPS = [ProcessingStepGroup(STEPS, task_name=TASK_NAME)]

process_pdq_39 = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes="pdq39-activity",
)
