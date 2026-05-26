from common.balanced_sample_tool import TheorySampler


sampler = TheorySampler('../Theories_dedup_by_SCI.csv')
print(sampler.a_range())
print(sampler.c_range())
sampler.draw_central_charge_histogram(0.1)
sampler.get_theory_stats().show(n=100, truncate=False)

balanced_sample = sampler.get_balanced_sample((0.5, 1.5), (0.5, 1.5), 50)
balanced_sample.get_theory_stats().show(n=100, truncate=False)

manual_sample = sampler.get_manual_sample([
    "SU5s1S1a1A1nf1",
    "H0H1",
    "SU2nf4",
    "SU6s1S1a1A1nf0",
    "Sp2adj1a2nf4"
], 50)
manual_sample.get_theory_stats().show(n=100, truncate=False)