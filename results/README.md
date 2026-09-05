# Frozen evaluation results

This folder presents public-safe aggregate results from the frozen experiment of
15 questions under three conditions (45 reports). The report-level generated text,
article passages and raw judge explanations are retained privately for authorised
academic review and are not published here.

## Main findings

- Conditions B and C passed the four-section structure check for 15/15 reports;
  Condition A passed for 10/15.
- Condition C recorded 92 visible citations and mean claim-verifier support of 0.611.
- Mean generation time was 24.57 seconds for A, 72.84 seconds for B and 212.54
  seconds for C, making C approximately 8.7 times slower than A.
- Mean pooled judge ratings were 4.133 for A, 4.182 for B and 3.969 for C.
- The pooled difference between conditions was not statistically significant
  (Friedman p = .109).
- Absolute agreement between the three judges was low (ICC(2,1) = .058), showing
  why the findings should not depend on one automated judge.

The automated cosine-based measures are not marks out of five and do not prove
factual correctness. The judge scores use a 1–5 rubric but remain model-based
assessments. Human evaluation is recommended as future work.

## Figures

### Three-judge comparison

![Three judges comparing A, B and C](figures/01_three_judge_comparison.png)

### Generation time

![Mean generation time](figures/02_generation_time.png)

### Structure checks

![Reports passing the structure check](figures/03_structure_checks.png)

### Automated measures

![Question relevance, evidence support and coverage](figures/04_automated_measures.png)

The `tables/` directory contains aggregate descriptive and statistical results.
