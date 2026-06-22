"""Cultural value dimensions used by the review analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """Definition for one cultural value dimension."""

    key: str
    label: str
    definition: str


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="social",
        label="Social value",
        definition=(
            "Spiritual: beliefs, myths, religions, legends, stories, and "
            "testimonials of past generations. Emotional (individual): memory "
            "and personal life experiences. Emotional (collective): cultural "
            "identity, motivation and pride, sense of place attachment, and "
            "communal value. Allegorical: objects or places representative of "
            "social hierarchy or status."
        ),
    ),
    Dimension(
        key="economic",
        label="Economic value",
        definition=(
            "Use: the function and utility of the asset, original or attributed. "
            "Non-use: an expired function that should remain through existence "
            "value, option value, and bequest value for future generations. "
            "Entertainment: the role the asset may have for the contemporary "
            "market, mainly tourism. Allegorical: meanings oriented to "
            "publicizing financial property."
        ),
    ),
    Dimension(
        key="political",
        label="Political value",
        definition=(
            "Educational: the educational role heritage assets may play when "
            "used for political targets, such as nation-birth myths or "
            "glorification of political leaders. Management: inclusion in past "
            "or present strategies and policies. Entertainment: dissemination "
            "of cultural awareness explored for political targets. Symbolic: "
            "emblematic, power, authority, and prosperous perceptions stemming "
            "from the heritage asset."
        ),
    ),
    Dimension(
        key="historic",
        label="Historic value",
        definition=(
            "Educational: heritage assets as a potential source for gaining "
            "knowledge about the past in the future. Historic-artistic: an "
            "object as one of few or unique testimonials of historic stylistic "
            "or artistic movements that are now part of history. "
            "Historic-conceptual: an object as one of few or unique "
            "testimonials retaining conceptual signs, such as architectural or "
            "urban-planning concepts, that are now part of history. Symbolic: "
            "the object has been part of or related to an important past event. "
            "Archaeological: connection with ancient civilizations."
        ),
    ),
    Dimension(
        key="aesthetical",
        label="Aesthetical value",
        definition=(
            "Artistic: an original product of creativity and imagination. "
            "Notable: a product of a creator, holding the creator's signature. "
            "Conceptual: integral materialization of conceptual intentions, "
            "implying a conceptual background. Evidential: an authentic exemplar "
            "of a decade or part of the history of art or architecture."
        ),
    ),
    Dimension(
        key="scientific",
        label="Scientific value",
        definition=(
            "Workmanship: original result of human labor and craftsmanship. "
            "Technological: skillfulness in techniques and materials, "
            "representing outstanding quality of work. Conceptual: integral "
            "materialization of conceptual intentions, implying a conceptual "
            "background."
        ),
    ),
    Dimension(
        key="age",
        label="Age value",
        definition=(
            "Workmanship: craftsmanship value oriented toward the production "
            "period. Existential: a piece of memory reflecting the passage and "
            "lives of past generations. Maturity: marks of time passage, such "
            "as patina, present on forms, components, and materials."
        ),
    ),
    Dimension(
        key="ecological",
        label="Ecological value",
        definition=(
            "Spiritual: harmony between the building and its natural or "
            "artificial environment. Essential: identification of ecological "
            "ideologies in design and construction. Existential: manufactured "
            "resources that can be reused, reprocessed, or recycled."
        ),
    ),
)

DIMENSION_KEYS: tuple[str, ...] = tuple(d.key for d in DIMENSIONS)


def dimension_definition_block() -> str:
    """Return a numbered English definition block for prompts."""

    return "\n".join(
        f"{index}. {dimension.label} ({dimension.key}): {dimension.definition}"
        for index, dimension in enumerate(DIMENSIONS, start=1)
    )


def empty_dimension_payload() -> dict[str, dict[str, object]]:
    """Return an empty payload for every dimension."""

    return {
        dimension.key: {"rationale": "", "evidence": []}
        for dimension in DIMENSIONS
    }
