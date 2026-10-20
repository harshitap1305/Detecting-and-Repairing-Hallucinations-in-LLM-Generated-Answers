"""
Unit tests for evaluate.py: metrics, drift, bootstrap CIs, and Wilcoxon testing.
"""
import unittest
from unittest.mock import patch

from src.evaluate import (
    bootstrap_confidence_interval,
    compute_detection_metrics,
    compute_levenshtein_distance,
    compute_normalized_levenshtein_similarity,
    compute_over_edit_rate,
    compute_preservation_rate,
    compute_rouge_l,
    compute_scr,
    compute_word_retention,
    evaluate_dataset,
    evaluate_with_independent_llm,
    paired_wilcoxon_test,
)
from src.schema import AnswerRecord, Claim


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.sample_record = AnswerRecord(
            id="eval_rec_01",
            source_dataset="manual",
            split="dev",
            question="When was Marie Curie born?",
            answer="Marie Curie was born in 1867 in Warsaw, Poland. She won two Nobel prizes.",
            claims=[
                Claim(
                    id="c1",
                    text="Marie Curie was born in 1867 in Warsaw, Poland.",
                    sentence_id=0,
                    check_worthy=True,
                    label="Supported",
                ),
                Claim(
                    id="c2",
                    text="Marie Curie won two Nobel prizes.",
                    sentence_id=1,
                    check_worthy=True,
                    label="Supported",
                ),
            ],
            repaired_answer="Marie Curie was born in 1867 in Warsaw, Poland. She won two Nobel prizes.",
            reliability_score_before=1.0,
            reliability_score_after=1.0,
        )

    def test_rouge_l(self):
        # Identical texts
        res = compute_rouge_l("hello world test", "hello world test")
        self.assertAlmostEqual(res["f1"], 1.0)
        self.assertAlmostEqual(res["precision"], 1.0)
        self.assertAlmostEqual(res["recall"], 1.0)

        # Disjoint texts
        res_disjoint = compute_rouge_l("apples oranges", "bananas grapes")
        self.assertAlmostEqual(res_disjoint["f1"], 0.0)

        # Partial overlap
        res_partial = compute_rouge_l("the quick brown fox", "the fast brown fox")
        self.assertGreater(res_partial["f1"], 0.5)

    def test_levenshtein(self):
        # Identical
        self.assertEqual(compute_levenshtein_distance("kitten", "kitten"), 0)
        self.assertAlmostEqual(compute_normalized_levenshtein_similarity("kitten", "kitten"), 1.0)

        # Classic kitten -> sitting dist is 3
        dist = compute_levenshtein_distance("kitten", "sitting")
        self.assertEqual(dist, 3)
        sim = compute_normalized_levenshtein_similarity("kitten", "sitting")
        self.assertAlmostEqual(sim, 1.0 - (3 / 7))

    def test_word_retention(self):
        orig = "The quick brown fox jumps"
        cand = "The brown fox leaps"
        ret = compute_word_retention(cand, orig)
        # 3 out of 5 words retained (the, brown, fox)
        self.assertAlmostEqual(ret, 0.6)

    def test_detection_metrics(self):
        preds = ["Supported", "Contradicted", "Contradicted", "Supported"]
        golds = ["Supported", "Contradicted", "Supported", "Unverifiable"]
        # Positives = {Contradicted, Unverifiable}
        # preds: [F, T, T, F]
        # golds: [F, T, F, T]
        # TP = 1 (item 2), FP = 1 (item 3), FN = 1 (item 4), TN = 1 (item 1)
        res = compute_detection_metrics(preds, golds)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tn"], 1)
        self.assertAlmostEqual(res["precision"], 0.5)
        self.assertAlmostEqual(res["recall"], 0.5)
        self.assertAlmostEqual(res["f1"], 0.5)

    def test_scr_computation(self):
        scr_val = compute_scr(self.sample_record)
        self.assertAlmostEqual(scr_val, 1.0)

        # Record with mixed claims
        rec_mixed = AnswerRecord(
            id="mixed",
            source_dataset="manual",
            question="Q",
            answer="A",
            claims=[
                Claim(id="1", text="T1", sentence_id=0, check_worthy=True, label="Supported"),
                Claim(id="2", text="T2", sentence_id=0, check_worthy=True, label="Contradicted"),
                Claim(id="3", text="T3", sentence_id=0, check_worthy=False),
            ],
        )
        self.assertAlmostEqual(compute_scr(rec_mixed), 0.5)

    def test_preservation_and_over_edit_rate(self):
        # Repaired answer identical to original -> 100% preservation, 0% over-edit
        pres = compute_preservation_rate(self.sample_record)
        over_edit = compute_over_edit_rate(self.sample_record)
        self.assertAlmostEqual(pres, 1.0)
        self.assertAlmostEqual(over_edit, 0.0)

        # Truncated repaired answer that drops a supported claim
        corrupted_record = AnswerRecord(
            id="corrupt",
            source_dataset="manual",
            question="Q",
            answer="Marie Curie was born in Warsaw. The Nobel Prize ceremony was held in Stockholm.",
            claims=[
                Claim(id="c1", text="Marie Curie was born in Warsaw.", sentence_id=0, label="Supported"),
                Claim(id="c2", text="The Nobel Prize ceremony was held in Stockholm.", sentence_id=1, label="Supported"),
            ],
            repaired_answer="Marie Curie was born in Warsaw.",  # Dropped Stockholm ceremony
        )
        pres_corr = compute_preservation_rate(corrupted_record)
        self.assertLess(pres_corr, 1.0)
        self.assertAlmostEqual(compute_over_edit_rate(corrupted_record), 1.0 - pres_corr)

    def test_bootstrap_confidence_interval(self):
        scores = [0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.87, 0.89]
        mean_val, low, high = bootstrap_confidence_interval(scores, n_bootstrap=500, seed=42)
        self.assertAlmostEqual(mean_val, sum(scores) / len(scores))
        self.assertLessEqual(low, mean_val)
        self.assertGreaterEqual(high, mean_val)

    def test_paired_wilcoxon_test(self):
        # System is systematically higher than baseline
        sys_scores = [0.90, 0.88, 0.95, 0.92, 0.89, 0.94, 0.91, 0.93, 0.96, 0.90]
        base_scores = [0.70, 0.65, 0.72, 0.68, 0.69, 0.71, 0.66, 0.70, 0.73, 0.67]
        res = paired_wilcoxon_test(sys_scores, base_scores)
        self.assertLess(res["p_value"], 0.05)
        self.assertTrue(res["is_significant_p05"])
        self.assertGreater(res["mean_difference"], 0.0)

        # Identical scores should produce p_value == 1.0 and not significant
        res_ident = paired_wilcoxon_test(sys_scores, sys_scores)
        self.assertFalse(res_ident["is_significant_p05"])

    def test_evaluate_dataset(self):
        dataset_records = [self.sample_record]
        results = evaluate_dataset(dataset_records, n_bootstrap=100)
        self.assertEqual(results["n_records"], 1)
        self.assertIn("scr", results)
        self.assertIn("preservation_rate", results)
        self.assertIn("lexical_similarity", results)
        self.assertAlmostEqual(results["scr"]["after"]["mean"], 1.0)
        self.assertAlmostEqual(results["preservation_rate"]["mean"], 1.0)

    @patch("src.evaluate.llm_call")
    def test_evaluate_with_independent_llm(self, mock_llm):
        mock_llm.return_value = '{"is_supported": true, "scr_estimate": 0.95, "hallucinated_phrases": [], "critique": "Factually accurate"}'
        res = evaluate_with_independent_llm(self.sample_record)
        self.assertTrue(res["is_supported"])
        self.assertAlmostEqual(res["scr_estimate"], 0.95)


if __name__ == "__main__":
    unittest.main()
