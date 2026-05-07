"""Tests for synthetic test-data generation via Ragas TestsetGenerator."""
from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MOCK_DOCS = [
    Document(
        page_content=(
            "The Selenium WebDriver Python course is a comprehensive programme designed for "
            "software testers and developers who want to master browser automation. The course "
            "contains 23 articles and 9 downloadable resources, covering everything from "
            "environment setup to advanced framework design.\n\n"
            "In the first section, students learn how to install Python, set up a virtual "
            "environment, and install Selenium using pip. The course explains how to download "
            "and configure browser drivers for Chrome, Firefox, and Edge, and how to use "
            "WebDriverManager to handle driver versions automatically.\n\n"
            "The second section covers element location strategies in depth. Students practise "
            "finding elements by ID, name, class name, tag name, link text, partial link text, "
            "CSS selector, and XPath. Detailed exercises show when to prefer CSS selectors over "
            "XPath and how to write robust, maintenance-friendly locators that survive minor UI "
            "changes.\n\n"
            "Synchronisation is addressed in the third section. Implicit waits, explicit waits "
            "using WebDriverWait and expected conditions, and fluent waits are all demonstrated "
            "with real-world examples. Students learn why Thread.sleep is an anti-pattern and "
            "how to eliminate flakiness caused by race conditions between the test and the page.\n\n"
            "Advanced topics include handling dropdowns with the Select class, interacting with "
            "checkboxes and radio buttons, uploading files, accepting and dismissing alerts, "
            "switching between browser windows and tabs, and working inside iframes. The course "
            "also covers taking full-page and element-level screenshots for failure documentation.\n\n"
            "The final section introduces the Page Object Model design pattern. Students refactor "
            "a monolithic test suite into a layered architecture with page classes, action methods, "
            "and a separate test layer. The course closes with integrating the suite into a pytest "
            "configuration, generating HTML reports with pytest-html, and running tests in a "
            "headless mode suitable for CI/CD pipelines."
        ),
        metadata={"source": "selenium_course.docx"},
    ),
    Document(
        page_content=(
            "The REST API Testing with Rest Assured Java course is aimed at QA engineers who "
            "need to validate backend services independently of the UI. It covers the full "
            "spectrum of API testing from manual exploration with Postman through to a fully "
            "automated Maven-based Rest Assured framework.\n\n"
            "The opening section explains the HTTP protocol: methods (GET, POST, PUT, PATCH, "
            "DELETE), status code families (2xx success, 3xx redirection, 4xx client error, "
            "5xx server error), headers, query parameters, path parameters, and the structure "
            "of JSON and XML response bodies. Students use Postman to send requests and "
            "inspect responses before writing a single line of code.\n\n"
            "The Rest Assured section begins with project setup in IntelliJ using Maven. "
            "Students learn the given-when-then BDD syntax and write their first assertions "
            "against status codes and response bodies using Hamcrest matchers. JSONPath and "
            "XmlPath expressions are used to extract nested values from complex response "
            "structures, and the course shows how to deserialise responses directly into "
            "Java POJOs using Jackson.\n\n"
            "Authentication is covered in a dedicated section: Basic Auth, Bearer token auth, "
            "OAuth 2.0 client credentials flow, and cookie-based session auth. Students build "
            "a reusable authentication utility that other test classes can call without "
            "duplicating credentials or token-fetch logic.\n\n"
            "Data-driven testing is introduced using TestNG DataProvider and an external Excel "
            "spreadsheet. The course shows how to parameterise request payloads so that one "
            "test method covers dozens of input scenarios. Contract testing with JSON Schema "
            "validation is also demonstrated, ensuring API responses conform to a declared "
            "schema even when the development team changes the payload structure.\n\n"
            "The course concludes with framework design: a base test class, a request "
            "specification builder, environment-specific configuration via properties files, "
            "and integration with a Jenkins pipeline that runs the suite on every pull request "
            "and publishes the Extent Reports HTML output as a build artefact."
        ),
        metadata={"source": "rest_api_course.docx"},
    ),
]

_EXPECTED_SAMPLE_COUNT = 2


@pytest.mark.data_creation
@pytest.mark.slow
@pytest.mark.asyncio
async def test_testset_generation(llm_wrapper, embeddings_wrapper) -> None:
    """
    Verify that TestsetGenerator produces the expected number of samples.
    In Ragas, this test ensures that the TestsetGenerator creates exactly the intended number of 
    evaluation samples based on the configured parameters.
    It validates that generation logic (e.g., chunking, sampling strategy) is functioning correctly.
    This helps prevent under- or over-generation, which can skew evaluation results.
    Passing the test confirms dataset consistency and reliability for downstream LLM metric evaluation.
    """
    from ragas.testset import TestsetGenerator  # local import — keeps 'Test*' name off module scope

    generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings_wrapper)
    # generate_with_langchain_docs returns EvaluationDataset at runtime; ragas stubs mis-annotate it as Executor
    dataset = generator.generate_with_langchain_docs(_MOCK_DOCS, testset_size=_EXPECTED_SAMPLE_COUNT)  # type: ignore[assignment]
    samples = dataset.to_list()  # type: ignore[attr-defined]

    logger.info("Generated %d test samples", len(samples))

    assert len(samples) == _EXPECTED_SAMPLE_COUNT, (
        f"Expected {_EXPECTED_SAMPLE_COUNT} samples, got {len(samples)}"
    )
    # Ragas >=0.4 uses 'user_input' / 'reference' in to_list() output, not 'question' / 'answer'
    for i, sample in enumerate(samples):
        assert "user_input" in sample, f"Sample {i} is missing 'user_input'"
        assert "reference" in sample, f"Sample {i} is missing 'reference'"
        assert sample["user_input"], f"Sample {i} has an empty user_input"
