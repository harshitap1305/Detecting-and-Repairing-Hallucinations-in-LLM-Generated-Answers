"""
Unit tests for demo/app.py: HTML formatting, word-level diff, and interactive execution.
"""
import unittest
from unittest.mock import patch

from demo.app import (
    format_claims_table,
    format_highlighted_text,
    format_metrics_banner,
    generate_word_diff_html,
    run_interactive_pipeline,
)
from src.schema import AnswerRecord, Claim, EvidencePassage


class TestDemoComponents(unittest.TestCase):
    def setUp(self):
        self.sample_record = AnswerRecord(
            id="demo_test_01",
            source_dataset="demo",
            split="dev",
            question="Where was Albert Einstein born?",
            answer="Albert Einstein was born in 1879 in Berlin. He developed relativity.",
            repaired_answer="Albert Einstein was born in 1879 in Ulm. He developed relativity.",
            reliability_score_before=0.50,
            reliability_score_after=1.00,
            repair_iterations=1,
            claims=[
                Claim(
                    id="demo_c1",
                    text="Albert Einstein was born in 1879.",
                    sentence_id=0,
                    check_worthy=True,
                    label="Supported",
                    label_rationale="Matches 1879 passage.",
                    evidence=[
                        EvidencePassage(
                            doc_id="einstein_0",
                            title="Albert Einstein",
                            text="Albert Einstein was born in Ulm in 1879.",
                            score=0.95,
                        )
                    ],
                ),
                Claim(
                    id="demo_c2",
                    text="Albert Einstein was born in Berlin.",
                    sentence_id=0,
                    check_worthy=True,
                    label="Contradicted",
                    label_rationale="Born in Ulm, not Berlin.",
                ),
            ],
        )

    def test_format_highlighted_text(self):
        html_out = format_highlighted_text(self.sample_record)
        self.assertIn("Contradicted", html_out)
        self.assertIn("Albert Einstein was born in 1879 in Berlin.", html_out)

    def test_generate_word_diff_html(self):
        orig = "Einstein was born in Berlin."
        rep = "Einstein was born in Ulm."
        diff_html = generate_word_diff_html(orig, rep)
        self.assertIn("<del", diff_html)
        self.assertIn("Berlin.", diff_html)
        self.assertIn("<ins", diff_html)
        self.assertIn("Ulm.", diff_html)

    def test_format_metrics_banner(self):
        banner_html = format_metrics_banner(self.sample_record)
        self.assertIn("50.0%", banner_html)
        self.assertIn("100.0%", banner_html)
        self.assertIn("+50.0%", banner_html)

    def test_format_claims_table(self):
        table_data = format_claims_table(self.sample_record)
        self.assertEqual(len(table_data), 2)
        self.assertEqual(table_data[0][0], "demo_c1")
        self.assertEqual(table_data[0][3], "Supported")
        self.assertEqual(table_data[0][5], "Albert Einstein")

    @patch("demo.app.run_pipeline")
    def test_run_interactive_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = self.sample_record
        q = "Where was Albert Einstein born?"
        a = "Albert Einstein was born in 1879 in Berlin. He developed relativity."

        high_html, diff_html, metrics_html, table_data = run_interactive_pipeline(q, a)
        self.assertIn("Contradicted", high_html)
        self.assertIn("<ins", diff_html)
        self.assertIn("Reliability Gain", metrics_html)
        self.assertEqual(len(table_data), 2)


if __name__ == "__main__":
    unittest.main()
